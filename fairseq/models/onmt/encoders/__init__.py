"""Module defining encoders."""
from .encoder import EncoderBase
from .transformer import TransformerEncoder
from .rnn_encoder import RNNEncoder
from .cnn_encoder import CNNEncoder
from .mean_encoder import MeanEncoder
from .audio_encoder import AudioEncoder
from .image_encoder import ImageEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
