from torch.nn import Module
from transformers import Wav2Vec2ForCTC


def wav2vec2_base(pretrained=True) -> Module:
    """Wav2vec 2.0 model ("Large" architecture),
    pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset
    (the combination of "train-clean-100", "train-clean-360", and "train-other-500"), not fine-tuned.

    Args:
        pretrained (bool, optional): Whether to load a pretrained (without fine-tuning) or to get randomly initialized model (not implemented yet). Defaults to True. Defaults to True.

    Raises:
        NotImplementedError: if pretrained = False

    Returns:
        Module: Wav2vec2 Base model
    """
    if pretrained is False:
        raise NotImplementedError(
            "Audio pretraining from scratch is not implemented yet"
        )
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base", return_dict=False)
    model.freeze_feature_extractor()
    return model


def wav2vec2_large(pretrained=True) -> Module:
    """Wav2vec 2.0 model ("Large" architecture),
    pre-trained on 960 hours of unlabeled audio from *LibriSpeech* dataset
    (the combination of "train-clean-100", "train-clean-360", and "train-other-500"), not fine-tuned.

        Args:
            pretrained (bool, optional): Whether to load a pretrained (without fine-tuning) or to get randomly initialized model (not implemented yet). Defaults to True.

        Raises:
            NotImplementedError: if pretrained = False

        Returns:
            Module: Wav2vec2 Large model
    """
    if pretrained is False:
        raise NotImplementedError(
            "Audio pretraining from scratch is not implemented yet"
        )
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large", return_dict=False)
    model.freeze_feature_extractor()
    return model
