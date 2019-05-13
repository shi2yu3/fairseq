"""Module defining decoders."""
from .decoder import DecoderBase, InputFeedRNNDecoder, \
    StdRNNDecoder
from .transformer import TransformerDecoder
from .cnn_decoder import CNNDecoder


str2dec = {"rnn": StdRNNDecoder, "ifrnn": InputFeedRNNDecoder,
           "cnn": CNNDecoder, "transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder", "StdRNNDecoder", "CNNDecoder",
           "InputFeedRNNDecoder", "str2dec"]
