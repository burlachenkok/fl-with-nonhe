import numpy as np
import pyarrow as pa
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
from datasets import load_dataset
from typing import List, Dict, Union, Optional


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


class FLLibrispeech:
    def __init__(
        self,
        exec_ctx,
        args,
        root,
        train=True,
        client_id=None,
    ):
        self.num_clients = 7
        if train:
            self.dataset = load_dataset(
                "librispeech_asr",
                split="train.clean.100+train.clean.360+train.other.500",
                cache_dir=root,
            )
        else:
            # Maybe add or remove validation.other
            self.dataset = load_dataset(
                "librispeech_asr",
                split="validation.clean",
                cache_dir=root,
            )
        # Adding index column for easier reference
        self.dataset = self.dataset.add_column(
            name="idx", column=np.arange(len(self.dataset))
        )
        # Removing unnecessary columns
        self.dataset = self.dataset.remove_columns(["chapter_id", "id", "file"])
        self.client_indices = self._get_client_indices()
        self.set_client(client_id)
        self.tokenizer: Wav2Vec2CTCTokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            "facebook/wav2vec2-base"
        )

    def __len__(self) -> int:
        return self.length

    def set_client(self, index=None):
        if index is None:
            self.client_id = None
            self.length = len(self.dataset)

        else:
            if index < 0 or index >= self.num_clients:
                raise ValueError("Number of clients is out of bounds.")
            self.client_id = index
            self.length = len(self.client_indices[self.client_id])

    def __getitem__(self, index):
        if self.client_id is None:
            actual_index = index
        else:
            actual_index = self.client_indices[self.client_id][index]

        audio, label = (
            self.dataset[actual_index]["audio"]["array"],
            self.dataset[actual_index]["text"],
        )

        return {
            "input_values": torch.from_numpy(audio),
            "labels": self.tokenizer(label)["input_ids"],
        }

    @property
    def _unique_speakers(self) -> List[int]:
        return list(set(self.dataset["speaker_id"]))

    def _get_client_indices(self) -> Dict[int, List]:
        """Retrieves indices from datasets corresponding to each client

        Returns:
            Dict[List]: Dictionary containing keys corresponding to client id with list of associated indices
        """
        # TODO This takes around 1-minute for whole train data, can we optimize it more?
        client_indices = {
            i: self.dataset.data.filter(
                pa.compute.is_in(
                    self.dataset.data["speaker_id"], value_set=pa.array(client)
                )
            )[3].to_pylist()
            for i, client in enumerate(
                self._divide_speakers_by_clients(
                    self._unique_speakers, self.num_clients
                )
            )
        }
        return client_indices

    @staticmethod
    def _divide_speakers_by_clients(
        unique_speakers: List[int], num_clients: int
    ) -> List[List[int]]:
        """Assigns speakers to clients

        Args:
            unique_speakers (List[int]): List of unique speakers in dataset
            num_clients (int): Number of Clients

        Returns:
            List[List[int]]: Returns list of speakers assigned to each client. Indices indicates client index and inner list contains speakers ids
        """
        num_elements_per_part = len(unique_speakers) // num_clients

        num_parts_with_extra_element = len(unique_speakers) % num_clients

        parts = []

        start = 0
        for i in range(num_clients):
            end = start + num_elements_per_part

            if i < num_parts_with_extra_element:
                end += 1

            parts.append(unique_speakers[start:end])

            start = end

        return parts

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
