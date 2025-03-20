from transformers import Wav2Vec2Processor
from torch.nn.functional import log_softmax
import numpy as np
from jiwer import wer
import torch

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")


def compute_wer(pred, labels):
    pred_logits = log_softmax(pred[0], dim=-1)
    pred_ids = torch.argmax(pred_logits, dim=-1)

    labels[labels == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(labels, group_tokens=False)
    word_err = wer(hypothesis=pred_str, reference=label_str)

    return word_err
