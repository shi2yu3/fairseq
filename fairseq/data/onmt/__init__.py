"""Module defining inputters.

Inputters implement the logic of transforming raw data to vectorized inputs,
e.g., from a line of text to a sequence of embeddings.
"""
from .inputter import \
    load_old_vocab, get_fields, OrderedIterator, \
    build_vocab, old_style_vocab, filter_example
from .dataset_base import Dataset
from .text_dataset import text_sort_key, TextDataReader
from .image_dataset import img_sort_key, ImageDataReader
from .audio_dataset import audio_sort_key, AudioDataReader
from .datareader_base import DataReaderBase


str2reader = {
    "text": TextDataReader, "img": ImageDataReader, "audio": AudioDataReader}
str2sortkey = {
    'text': text_sort_key, 'img': img_sort_key, 'audio': audio_sort_key}


__all__ = ['Dataset', 'load_old_vocab', 'get_fields', 'DataReaderBase',
           'filter_example', 'old_style_vocab',
           'build_vocab', 'OrderedIterator',
           'text_sort_key', 'img_sort_key', 'audio_sort_key',
           'TextDataReader', 'ImageDataReader', 'AudioDataReader']
