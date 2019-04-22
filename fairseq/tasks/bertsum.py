# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

from fairseq import options, utils
from fairseq.data import (
    BertDictionary,
    BertDataset
)

from . import FairseqTask, register_task


@register_task('bert')
class BertTask(FairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        src_dict (Dictionary): dictionary for the source language
        tgt_dict (Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad', default='False', type=str, metavar='BOOL',
                            help='pad the sentence on the left')
        parser.add_argument('--max-positions', default=512, type=int, metavar='N',
                            help='max number of tokens in the sequence')
        # fmt: on

    @staticmethod
    def load_pretrained_model(path, src_dict_path, tgt_dict_path, arg_overrides=None):
        model = utils.load_checkpoint_to_cpu(path)
        args = model['args']
        state_dict = model['model']
        args = utils.override_model_args(args, arg_overrides)
        src_dict = Dictionary.load(src_dict_path)
        tgt_dict = Dictionary.load(tgt_dict_path)
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()

        task = TranslationTask(args, src_dict, tgt_dict)
        model = task.build_model(args)
        model.upgrade_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        return model

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dict = dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        args.left_pad = options.eval_bool(args.left_pad)

        # load dictionaries
        dictionary = BertDictionary.load(os.path.join(args.data[0], 'vocab.txt'))
        print('| dictionary: {} types'.format(len(dictionary)))

        return cls(args, dictionary)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, data_path):
            filename = os.path.join(data_path, split)
            if self.args.raw_text and IndexedRawTextDataset.exists(filename):
                return True
            elif not self.args.raw_text and IndexedDataset.exists(filename):
                return True
            return False

        def indexed_dataset(path, dictionary):
            if self.args.raw_text:
                return IndexedRawTextDataset(path, dictionary)
            elif IndexedDataset.exists(path):
                if self.args.lazy_load:
                    return IndexedDataset(path, fix_lua_indexing=True)
                else:
                    return IndexedCachedDataset(path, fix_lua_indexing=True)
            return None

        datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                if split_exists(split_k, data_path):
                    prefix = os.path.join(data_path, split_k)
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                datasets.append(indexed_dataset(prefix, self.dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(datasets[-1])))

                if not combine:
                    break

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            sample_ratios = [1] * len(datasets)
            sample_ratios[0] = self.args.upsample_primary
            dataset = ConcatDataset(datasets, sample_ratios)

        self.datasets[split] = BertDataset(
            dataset, dataset.sizes, self.dict,
            left_pad=self.args.left_pad,
            max_positions=self.args.max_positions,
        )

    def build_dataset_for_inference(self, tokens, lengths):
        return BertDataset(tokens, lengths, self.dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return self.args.max_positions

    @property
    def dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.dict
