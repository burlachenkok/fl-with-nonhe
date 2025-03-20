import numpy as np
import pyarrow as pa
import torch
import os
import tarfile
import requests
import hashlib
import time
import librosa
import glob
import re
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from torch.utils.data import Dataset
from typing import List, Dict, Union, Optional
from tqdm.auto import tqdm


class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    def __init__(self, processor, padding=True):
        self.processor: Wav2Vec2Processor = processor
        self.padding: Union[bool, str] = padding

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_values": feature["input_values"]} for feature in features
        ]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels
        if self.processor.feature_extractor.return_attention_mask:
            return (
                {
                    "input_values": batch["input_values"],
                    "attention_mask": batch["attention_mask"],
                },
                batch["labels"],
            )

        return {**batch}


def download_librilight(destination):
    def download_and_extract(url, destination, expected_checksum, parent_folder=None):
        filename = os.path.join(destination, url.split("/")[-1])

        def calculate_checksum(file_path):
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as file:
                for chunk in iter(lambda: file.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()

        def files_extracted(destination, filename, parent_folder=None):
            archive_name = os.path.basename(filename)
            folder_name = re.sub(r"\.(tar\.gz|tgz)$", "", archive_name)
            if parent_folder:
                folder_name = os.path.join(parent_folder, folder_name)
                print(f"FOLDER NAME {folder_name}")
            return os.path.exists(os.path.join(destination, folder_name))

        while (
            not os.path.exists(filename)
            or calculate_checksum(filename) != expected_checksum
        ):
            max_retries = 5
            base_sleep_time = 5  # seconds

            for attempt in range(max_retries):
                try:
                    print(f"Downloading {url} to {filename} (Attempt {attempt + 1})...")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()

                    total_size = int(response.headers.get("content-length", 0))
                    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)

                    with open(filename, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                            progress_bar.update(len(chunk))

                    progress_bar.close()

                    if calculate_checksum(filename) == expected_checksum:
                        print(f"Download successful.")
                        break

                    print(f"File checksum does not match. Retrying download.")
                except requests.exceptions.RequestException as error:
                    if attempt + 1 == max_retries:
                        print("Max retries reached. Download failed.")
                        raise error
                    else:
                        sleep_time = base_sleep_time * (2**attempt)
                        print(
                            f"Error occurred: {error}. Retrying in {sleep_time} seconds..."
                        )
                        time.sleep(sleep_time)

        # Extract files
        if not files_extracted(destination, filename, parent_folder):
            with tarfile.open(filename, "r:gz") as tar:
                # Adding progress bar for file extraction
                members = tar.getmembers()
                progress_bar = tqdm(total=len(members), unit="files")

                for member in members:
                    tar.extract(member, destination)
                    progress_bar.update(1)

            progress_bar.close()
            print("Extraction complete.")
        else:
            print(f"Files already extracted. Using existing files for {filename}.")

    if not os.path.exists(destination):
        os.makedirs(destination)

    librilight_url = (
        "https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz"
    )
    librilight_checksum = "7f83024cb1334bfa372d1af2c75c3a77"

    dev_clean_url = "http://www.openslr.org/resources/12/dev-clean.tar.gz"
    dev_clean_checksum = "42e2234ba48799c1f50f24a7926300a1"

    dev_other_url = "http://www.openslr.org/resources/12/dev-other.tar.gz"
    dev_other_checksum = "c8d0bcc9cca99d4f8b62fcc847357931"

    download_and_extract(librilight_url, destination, librilight_checksum)
    download_and_extract(
        dev_clean_url, destination, dev_clean_checksum, parent_folder="LibriSpeech"
    )
    download_and_extract(
        dev_other_url, destination, dev_other_checksum, parent_folder="LibriSpeech"
    )


class FLLibrilight(Dataset):
    def __init__(
        self,
        exec_ctx,
        args,
        root,
        train=True,
        client_id=None,
    ):
        download_librilight(root)
        self.train = train
        ## Setting split for train
        if self.train:
            self.split = "10h"
            assert self.split in [
                "9h",
                "1h",
                "10h",
            ], "Invalid split value. Allowed values are '9h', '1h', and '10h'."
            self.base_dir = os.path.join(root, "librispeech_finetuning")

        ## Setting split for dev
        else:
            self.split = "dev-clean"
            assert self.split in [
                "dev-clean",
                "dev-other",
                "all",
            ], "Invalid split value. Allowed values are 'dev-clean', 'dev-other', and 'all'."
            self.base_dir = os.path.join(root, "LibriSpeech")

        self.num_clients = 5
        assert self.num_clients <= 24, "Number of clients higher than 24 is not allowed"

        self.data = []
        self.client_indices = [[] for _ in range(self.num_clients)]

        # Load data and divide among clients
        self._load_data()
        self.client_id = client_id
        self.set_client(self.client_id)
        self.tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "facebook/wav2vec2-base",
        )

    def _load_data(self):
        if self.train:
            if self.split == "10h":
                split_subdirectories = ["1h", "9h"]
            else:
                split_subdirectories = [self.split]
        else:
            if self.split == "all":
                split_subdirectories = ["dev-clean", "dev-other"]
            else:
                split_subdirectories = [self.split]
        trans_files = []
        audio_files = []
        for subdir in split_subdirectories:
            split_dir = os.path.join(self.base_dir, subdir)

            # Search for all trans.txt files
            trans_files.extend(
                glob.glob(os.path.join(split_dir, "**/*.trans.txt"), recursive=True)
            )

            # Find all Flac files
            audio_files.extend(
                glob.glob(os.path.join(split_dir, "**/*.flac"), recursive=True)
            )

        labels = {}
        for trans_file in trans_files:
            self._load_single_trans_file(trans_file, labels)

        # Split speakers among clients
        speaker_ids = list(
            set(
                [
                    os.path.splitext(os.path.basename(f))[0].split("-")[0]
                    for f in audio_files
                ]
            )
        )
        client_assignments = {
            speaker_id: i % self.num_clients for i, speaker_id in enumerate(speaker_ids)
        }

        for audio_file in audio_files:
            file_id = os.path.splitext(os.path.basename(audio_file))[0]
            speaker_id = file_id.split("-")[0]
            client_id = client_assignments[speaker_id]
            self.client_indices[client_id].append(len(self.data))

            self.data.append(
                {
                    "audio_file": audio_file,
                    "label": labels[file_id],
                    "speaker_id": speaker_id,
                }
            )

    def _load_single_trans_file(self, trans_file, labels):
        with open(trans_file, "r") as f:
            for line in f:
                file_id, label_text = line.strip().split(" ", 1)
                labels[file_id] = label_text

    def set_client(self, index=None):
        if index is None:
            self.client_id = None
            self.length = len(self.data)

        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError("Number of clients is out of bounds.")
            self.client_id = index
            self.length = len(self.client_indices[self.client_id])
            self.active_client_indices = self.client_indices[self.client_id]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx):
        if self.client_id is not None:
            idx = self.active_client_indices[idx]

        audio, sample_rate = librosa.load(self.data[idx]["audio_file"])

        label = self.data[idx]["label"]

        return {
            "input_values": torch.from_numpy(audio),
            "labels": self.tokenizer(label)["input_ids"],
        }

    @staticmethod
    def get_collate_func(model_name) -> DataCollatorCTCWithPadding:
        if model_name == "wav2vec2_base":
            collator = DataCollatorCTCWithPadding(
                processor=Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base"),
                padding=True,
            )
            return collator

        collator = DataCollatorCTCWithPadding(
            processor=Wav2Vec2Processor.from_pretrained(
                "facebook/wav2vec2-base", return_attention_mask=True
            ),
            padding=True,
        )
        return collator
