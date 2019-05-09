# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os

import torch
import numpy as np

from fairseq.data import (
    ConcatDataset,
    Dictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    MonolingualDataset,
    TokenBlockDataset,
    TransformEosDataset,
    TruncatedDictionary,
)

from . import FairseqTask, register_task


@register_task('enas')
class EnasTask(FairseqTask):
    """

    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='path to data directory')
        parser.add_argument('--reset_output_dir', action='store_true',
                            help='Delete output_dir if exists.')
        parser.add_argument('--output_dir', default='', type=str)
        parser.add_argument('--data_format', default='NHWC', type=str,
                            help='NHWC or NCWH')
        parser.add_argument('--search_for', default=None, type=str,
                            help='Must be [macro|micro]')
        parser.add_argument('--search_hparam', action='store_true',
                            help='Search optimal parameters or not.')
        parser.add_argument('--dataset', default='sst', type=str,
                            help='dataset for evaluation')
        parser.add_argument('--embedding_model', default='word2vec', type=str,
                            help='word2vec or glove')
        parser.add_argument('--child_lr_decay_scheme', default='exponential', type=str,
                            help='Strategy to decay learning rate, '
                                 'must be [cosine, noam, exponential, auto]')

        parser.add_argument('--child_lr_decay_epoch_multiplier', default=1, type=int)

        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--max_input_length', default=50, type=int)
        parser.add_argument('--class_num', default=5, type=int)
        parser.add_argument('--pool_step', default=3, type=int)

        parser.add_argument('--num_epochs', type=int, default=300)
        parser.add_argument('--child_lr_dec_start', type=int, default=0)
        parser.add_argument('--child_lr_dec_every', type=int, default=100)
        parser.add_argument('--child_lr_warmup_steps', type=int, default=100)
        parser.add_argument('--child_lr_model_d', type=int, default=300)
        parser.add_argument('--child_num_layers', type=int, default=5)
        parser.add_argument('--child_num_cells', type=int, default=5)
        parser.add_argument('--child_filter_size', type=int, default=5)
        parser.add_argument('--child_out_filters', type=int, default=48)
        parser.add_argument('--child_out_filters_scale', type=int, default=1)
        parser.add_argument('--child_num_branches', type=int, default=4)
        parser.add_argument('--child_lr_T_0', type=int, default=None,
                            help='for lr schedule')
        parser.add_argument('--child_lr_T_mul', type=int, default=None,
                            help='for lr schedule')
        parser.add_argument('--min_count', type=int, default=0)
        parser.add_argument('--train_ratio', type=float, default=0.5)
        parser.add_argument('--valid_ratio', type=float, default=0.1)
        parser.add_argument('--child_grad_bound', type=float, default=5.0,
                     help='Gradient clipping')
        parser.add_argument('--child_lr', type=float, default=0.1)
        parser.add_argument('--child_lr_dec_rate', type=float, default=0.1)
        parser.add_argument('--child_keep_prob', type=float, default=0.5)
        parser.add_argument('--lstm_x_keep_prob', type=float, default=0.5)
        parser.add_argument('--lstm_h_keep_prob', type=float, default=0.8)
        parser.add_argument('--lstm_o_keep_prob', type=float, default=0.5)
        parser.add_argument('--embed_keep_prob', type=float, default=0.8)
        parser.add_argument('--attention_keep_prob', type=float, default=1.0)
        parser.add_argument('--child_l2_reg', type=float, default=1e-4)
        parser.add_argument('--child_lr_max', type=float, default=None,
                     help='for lr schedule')
        parser.add_argument('--child_lr_min', type=float, default=None,
                     help='for lr schedule')
        parser.add_argument('--child_fixed_arc', default=None, type=str)
        parser.add_argument('--child_optim_algo', default='momentum', type=str)
        parser.add_argument('--multi_path', action='store_true',
                            help='Search for multiple path in the architecture')
        parser.add_argument('--is_binary', action='store_true',
                            help='binary label for sst dataset')
        parser.add_argument('--positional_encoding', action='store_true',
                            help='use positional encoding on attention input')
        parser.add_argument('--input_positional_encoding', action='store_true',
                            help='use positional encoding as input')
        parser.add_argument('--is_sinusolid', action='store_true',
                            help='use sinusolid as positional encoding')
        parser.add_argument('--var_rec', action='store_true',
                            help='varational_recurrent')
        parser.add_argument('--skip_concat', action='store_true',
                            help='use concat in skip connection')
        parser.add_argument('--all_layer_output', action='store_true',
                            help='use all layer as output')
        parser.add_argument('--num_last_layer_output', type=int, default=0,
                            help='last n layers as output, 0 for all')
        parser.add_argument('--output_linear_combine', action='store_true',
                            help='linear combine of output layers')
        parser.add_argument('--is_debug', action='store_true',
                            help='whether print debug info')
        parser.add_argument('--is_mask', action='store_true',
                            help='whether apply mask')
        parser.add_argument('--is_output_attention', action='store_true',
                            help='apply attention before softmax output')
        parser.add_argument('--field_embedding', action='store_true',
                            help='whether use field embedding in attention')
        parser.add_argument('--input_field_embedding', action='store_true',
                            help='whether use field embedding in input')
        parser.add_argument('--sliding_window', action='store_true',
                            help='use sliding window as input')
        parser.add_argument('--fixed_seed', action='store_true',
                            help='whether fix the random seed')

        parser.add_argument('--controller_lr', type=float, default=1e-3)
        parser.add_argument('--controller_lr_dec_rate', type=float, default=1.0)
        parser.add_argument('--controller_keep_prob', type=float, default=0.5)
        parser.add_argument('--controller_l2_reg', type=float, default=0.0)
        parser.add_argument('--controller_bl_dec', type=float, default=0.99)
        parser.add_argument('--controller_tanh_constant', type=float, default=None)
        parser.add_argument('--controller_op_tanh_reduce', type=float, default=1.0)
        parser.add_argument('--controller_temperature', type=float, default=None)
        parser.add_argument('--controller_entropy_weight', type=float, default=None)
        parser.add_argument('--controller_skip_target', type=float, default=0.8)
        parser.add_argument('--controller_skip_weight', type=float, default=0.0)
        parser.add_argument('--controller_train_steps', type=int, default=50)
        parser.add_argument('--controller_train_every', type=int, default=2,
                            help='train the controller after this number of epochs')
        parser.add_argument('--controller_training', action='store_false')

        parser.add_argument('--log_every', type=int, default=50,
                            help='How many steps to log')
        parser.add_argument('--eval_every_epochs', type=int, default=1,
                            help='How many epochs to eval')
        # fmt: on

    def __init__(self, args, dictionary, output_dictionary, targets=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.output_dictionary = output_dictionary

        if targets is None:
            targets = ['future']
        self.targets = targets

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        dictionary = None
        output_dictionary = None
        if args.data:
            dictionary = Dictionary.load(os.path.join(args.data, 'dict.txt'))
            print('| dictionary: {} types'.format(len(dictionary)))
            output_dictionary = dictionary
            if args.output_dictionary_size >= 0:
                output_dictionary = TruncatedDictionary(dictionary, args.output_dictionary_size)

        # upgrade old checkpoints
        if hasattr(args, 'exclude_self_target'):
            args.self_target = not args.exclude_self_target

        targets = []
        if getattr(args, 'self_target', False):
            targets.append('self')
        if getattr(args, 'future_target', False):
            targets.append('future')
        if getattr(args, 'past_target', False):
            targets.append('past')
        if len(targets) == 0:
            # standard language modeling
            targets = ['future']

        return cls(args, dictionary, output_dictionary, targets=targets)

    def build_model(self, args):
        model = super().build_model(args)

        for target in self.targets:
            if target not in model.supported_targets:
                raise ValueError('Unsupported language modeling target: {}'.format(target))

        return model

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        loaded_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else '')
            path = os.path.join(self.args.data, split_k)

            if self.args.raw_text and IndexedRawTextDataset.exists(path):
                ds = IndexedRawTextDataset(path, self.dictionary)
            elif not self.args.raw_text and IndexedDataset.exists(path):
                if self.args.lazy_load:
                    ds = IndexedDataset(path, fix_lua_indexing=True)
                else:
                    ds = IndexedCachedDataset(path, fix_lua_indexing=True)
            else:
                if k > 0:
                    break
                else:
                    raise FileNotFoundError('Dataset not found: {} ({})'.format(split, self.args.data))

            loaded_datasets.append(
                TokenBlockDataset(
                    ds, ds.sizes, self.args.tokens_per_sample,
                    pad=self.dictionary.pad(), eos=self.dictionary.eos(),
                    break_mode=self.args.sample_break_mode, include_targets=True,
                )
            )

            print('| {} {} {} examples'.format(self.args.data, split_k, len(loaded_datasets[-1])))

            if not combine:
                break

        if len(loaded_datasets) == 1:
            dataset = loaded_datasets[0]
            sizes = dataset.sizes
        else:
            dataset = ConcatDataset(loaded_datasets)
            sizes = np.concatenate([ds.sizes for ds in loaded_datasets])

        add_eos_for_other_targets = self.args.sample_break_mode is not None and self.args.sample_break_mode != 'none'

        self.datasets[split] = MonolingualDataset(
            dataset, sizes, self.dictionary, self.output_dictionary,
            add_eos_for_other_targets=add_eos_for_other_targets, shuffle=True,
            targets=self.targets,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return TransformEosDataset(
            MonolingualDataset(
                TokenBlockDataset(
                    src_tokens,
                    src_lengths,
                    block_size=None,
                    pad=self.source_dictionary.pad(),
                    eos=self.source_dictionary.eos(),
                    break_mode='eos',
                    include_targets=False,
                ),
                src_lengths,
                self.source_dictionary,
                self.target_dictionary,
                add_eos_for_other_targets=False,
                shuffle=False,
            ),
            eos=self.source_dictionary.eos(),
            # remove EOS since this will be used as a prefix for generation
            remove_eos_from_src=True,
            has_target=False,
        )

    def inference_step(self, generator, models, sample, prefix_tokens=None):
        with torch.no_grad():
            if prefix_tokens is None:
                # note: EOS has already been removed in build_dataset_for_inference
                prefix_tokens = sample['net_input']['src_tokens']
            return generator.generate(models, sample, prefix_tokens=prefix_tokens)

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.dictionary

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.output_dictionary
