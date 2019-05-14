# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import itertools
import os
import glob

from fairseq import options, utils
from fairseq.data import (
    ConcatDataset,
    data_utils,
    OpenNMTDictionary,
    IndexedCachedDataset,
    IndexedDataset,
    IndexedRawTextDataset,
    LanguagePairDataset,
)
from fairseq.onmt_utils import parse


from . import FairseqTask, register_task


@register_task('summarization')
class SummarizationTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', nargs='+', help='path(s) to data directorie(s)')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language')
        parser.add_argument('--lazy-load', action='store_true',
                            help='load the dataset lazily')
        parser.add_argument('--raw-text', action='store_true',
                            help='load raw text dataset')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')

        # Init options
        parser.add_argument('--param_init', '-param_init', type=float, default=0.1,
                  help="Parameters are initialized over uniform distribution "
                       "with support (-param_init, param_init). "
                       "Use 0 to not use initialization")
        parser.add_argument('--param_init_glorot', '-param_init_glorot', action='store_true',
                  help="Init parameters with xavier_uniform. "
                       "Required for transformer.")

        # = '--restore-file'
        # parser.add_argument('--train_from', '-train_from', default='', type=str,
        #           help="If training from a checkpoint then this is the "
        #                "path to the pretrained model's state_dict.")
        # = '--reset-optimizer'
        # parser.add_argument('--reset_optim', '-reset_optim', default='none',
        #           choices=['none', 'all', 'states', 'keep_states'],
        #           help="Optimization resetter when train_from.")

        # Pretrained word vectors
        # parser.add_argument('--pre_word_vecs_enc', '-pre_word_vecs_enc',
        #           help="If a valid path is specified, then this will load "
        #                "pretrained word embeddings on the encoder side. "
        #                "See README for specific formatting instructions.")
        # parser.add_argument('--pre_word_vecs_dec', '-pre_word_vecs_dec',
        #           help="If a valid path is specified, then this will load "
        #                "pretrained word embeddings on the decoder side. "
        #                "See README for specific formatting instructions.")
        # # Fixed word vectors
        # parser.add_argument('--fix_word_vecs_enc', '-fix_word_vecs_enc',
        #           action='store_true',
        #           help="Fix word embeddings on the encoder side.")
        # parser.add_argument('--fix_word_vecs_dec', '-fix_word_vecs_dec',
        #           action='store_true',
        #           help="Fix word embeddings on the decoder side.")

        # Optimization options
        # = '--max-sentences' or '--batch-size'
        # parser.add_argument('--batch_size', '-batch_size', type=int, default=64,
        #           help='Maximum batch size for training')
        # = '--max-tokens' or '--max-sentences'
        # parser.add_argument('--batch_type', '-batch_type', default='sents',
        #           choices=["sents", "tokens"],
        #           help="Batch grouping for batch_size. Standard "
        #                "is sents. Tokens will do dynamic batching")
        # = '--sentence-avg'
        # parser.add_argument('--normalization', '-normalization', default='sents',
        #           choices=["sents", "tokens"],
        #           help='Normalization method of the gradient.')
        # = '--update-freq'
        # parser.add_argument('--accum_count', '-accum_count', type=int, nargs='+',
        #           default=[1],
        #           help="Accumulate gradient this many times. "
        #                "Approximately equivalent to updating "
        #                "batch_size * accum_count batches at once. "
        #                "Recommended for Transformer.")
        # parser.add_argument('--accum_steps', '-accum_steps', type=int, nargs='+',
        #           default=[0], help="Steps at which accum_count values change")
        # = '--validate-interval'
        # parser.add_argument('--valid_steps', '-valid_steps', type=int, default=10000,
        #           help='Perfom validation every X steps')
        # = '--max-sentences-valid'
        # parser.add_argument('--valid_batch_size', '-valid_batch_size', type=int, default=32,
        #           help='Maximum batch size for validation')
        parser.add_argument('--max_generator_batches', '-max_generator_batches',
                  type=int, default=32,
                  help="Maximum batches of words in a sequence to run "
                       "the generator on in parallel. Higher is faster, but "
                       "uses more memory. Set to 0 to disable.")
        # = '--max-epoch' and '--max-update'
        # parser.add_argument('--train_steps', '-train_steps', type=int, default=100000,
        #           help='Number of training steps')
        # parser.add_argument('--single_pass', '-single_pass', action='store_true',
        #           help="Make a single pass over the training dataset.")
        # parser.add_argument('--epochs', '-epochs', type=int, default=0,
        #           help='Deprecated epochs see train_steps')
        # = '--no-early-stop'
        # parser.add_argument('--early_stopping', '-early_stopping', type=int, default=0,
        #           help='Number of validation steps without improving.')
        # parser.add_argument('--early_stopping_criteria', '-early_stopping_criteria',
        #           nargs="*", default=None,
        #           help='Criteria to use for early stopping.')
        # = '--optimizer'
        # parser.add_argument('--optim', '-optim', default='sgd',
        #           choices=['sgd', 'adagrad', 'adadelta', 'adam',
        #                    'sparseadam', 'adafactor', 'fusedadam'],
        #           help="Optimization method.")
        # -> optim/adagrad.py
        # parser.add_argument('--adagrad_accumulator_init', '-adagrad_accumulator_init',
        #           type=float, default=0,
        #           help="Initializes the accumulator values in adagrad. "
        #                "Mirrors the initial_accumulator_value option "
        #                "in the tensorflow adagrad (use 0.1 for their default).")
        # = '--clip-norm'
        # parser.add_argument('--max_grad_norm', '-max_grad_norm', type=float, default=5,
        #           help="If the norm of the gradient vector exceeds this, "
        #                "renormalize it to have the norm equal to "
        #                "max_grad_norm")
        # parser.add_argument('--dropout', '-dropout', type=float, default=0.3,
        #                     help="Dropout probability; applied in LSTM stacks.")
        parser.add_argument('--truncated_decoder', '-truncated_decoder', type=int, default=0,
                  help="Truncated bptt.")
        # = '--adam-betas' (optim/adam.py)
        # parser.add_argument('--adam_beta1', '-adam_beta1', type=float, default=0.9,
        #           help="The beta1 parameter used by Adam. "
        #                "Almost without exception a value of 0.9 is used in "
        #                "the literature, seemingly giving good results, "
        #                "so we would discourage changing this value from "
        #                "the default without due consideration.")
        # parser.add_argument('--adam_beta2', '-adam_beta2', type=float, default=0.999,
        #           help='The beta2 parameter used by Adam. '
        #                'Typically a value of 0.999 is recommended, as this is '
        #                'the value suggested by the original paper describing '
        #                'Adam, and is also the value adopted in other frameworks '
        #                'such as Tensorflow and Kerras, i.e. see: '
        #                'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
        #                'Optimizer or '
        #                'https://keras.io/optimizers/ . '
        #                'Whereas recently the paper "Attention is All You Need" '
        #                'suggested a value of 0.98 for beta2, this parameter may '
        #                'not work well for normal models / default '
        #                'baselines.')
        # = '--label-smoothing' (criterions/label_smoothed_cross_entropy.py)
        # parser.add_argument('--label_smoothing', '-label_smoothing', type=float, default=0.0,
        #           help="Label smoothing value epsilon. "
        #                "Probabilities of all non-true labels "
        #                "will be smoothed by epsilon / (vocab_size - 1). "
        #                "Set to zero to turn off label smoothing. "
        #                "For more detailed information, see: "
        #                "https://arxiv.org/abs/1512.00567")
        # = '--weight-decay' ??
        # parser.add_argument('--average_decay', '-average_decay', type=float, default=0,
        #           help="Moving average decay. "
        #                "Set to other than 0 (e.g. 1e-4) to activate. "
        #                "Similar to Marian NMT implementation: "
        #                "http://www.aclweb.org/anthology/P18-4020 "
        #                "For more detail on Exponential Moving Average: "
        #                "https://en.wikipedia.org/wiki/Moving_average")
        # parser.add_argument('--average_every', '-average_every', type=int, default=1,
        #           help="Step for moving average. "
        #                "Default is every update, "
        #                "if -average_decay is set.")

        # learning rate
        # = '--lr'
        # parser.add_argument('--learning_rate', '-learning_rate', type=float, default=1.0,
        #           help="Starting learning rate. "
        #                "Recommended settings: sgd = 1, adagrad = 0.1, "
        #                "adadelta = 1, adam = 0.001")
        # = '--lr-shrink' ??
        # parser.add_argument('--learning_rate_decay', '-learning_rate_decay',
        #           type=float, default=0.5,
        #           help="If update_learning_rate, decay learning rate by "
        #                "this much if steps have gone past "
        #                "start_decay_steps")
        # = '--lr-scheduler'
        # parser.add_argument('--start_decay_steps', '-start_decay_steps',
        #           type=int, default=50000,
        #           help="Start decaying every decay_steps after "
        #                "start_decay_steps")
        # parser.add_argument('--decay_steps', '-decay_steps', type=int, default=10000,
        #           help="Decay every decay_steps")

        # = '--lr-scheduler'
        # parser.add_argument('--decay_method', '-decay_method', type=str, default="none",
        #           choices=['noam', 'noamwd', 'rsqrt', 'none'],
        #           help="Use a custom decay rate.")
        # = '--warmup-updates' (optim/lr_scheduler/*.py)
        # parser.add_argument('--warmup_steps', '-warmup_steps', type=int, default=4000,
        #           help="Number of warmup steps for custom decay.")

        # = '--log-interval'
        # parser.add_argument('--report_every', '-report_every', type=int, default=50,
        #           help="Print stats at this interval.")
        # parser.add_argument('--log_file', '-log_file', type=str, default="",
        #           help="Output logs to a file under this path.")
        # parser.add_argument('--log_file_level', '-log_file_level', type=str,
        #           action=StoreLoggingLevelAction,
        #           choices=StoreLoggingLevelAction.CHOICES,
        #           default="0")
        # parser.add_argument('--exp_host', '-exp_host', type=str, default="",
        #           help="Send logs to this crayon server.")
        # parser.add_argument('--exp', '-exp', type=str, default="",
        #           help="Name of the experiment for logging.")
        # Use TensorboardX for visualization during training
        # parser.add_argument('--tensorboard', '-tensorboard', action="store_true",
        #           help="Use tensorboardX for visualization during training. "
        #                "Must have the library tensorboardX.")
        # = '--tensorboard-logdir'
        # parser.add_argument("--tensorboard_log_dir", "-tensorboard_log_dir",
        #           type=str, default="runs/onmt",
        #           help="Log directory for Tensorboard. "
        #                "This is also the name of the run.")

        # Options most relevant to speech
        # parser.add_argument('--sample_rate', '-sample_rate', type=int, default=16000,
        #           help="Sample rate.")
        # parser.add_argument('--window_size', '-window_size', type=float, default=.02,
        #           help="Window size for spectrogram in seconds.")

        # Option most relevant to image input
        # parser.add_argument('--image_channel_size', '-image_channel_size',
        #           type=int, default=3, choices=[3, 1],
        #           help="Using grayscale image can training "
        #                "model faster and smaller")
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

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """

        def opennmt_compatible(args):
            args.epochs = 0
            return args
        args = opennmt_compatible(args)

        parse.ArgumentParser.validate_train_opts(args)
        parse.ArgumentParser.update_model_opts(args)
        parse.ArgumentParser.validate_model_opts(args)

        args.left_pad_source = options.eval_bool(args.left_pad_source)
        args.left_pad_target = options.eval_bool(args.left_pad_target)

        # find language pair automatically
        if args.source_lang is None or args.target_lang is None:
            args.source_lang, args.target_lang = data_utils.infer_language_pair(args.data[0])
        if args.source_lang is None or args.target_lang is None:
            raise Exception('Could not infer language pair, please provide it explicitly')

        # load dictionaries
        src_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.source_lang)))
        tgt_dict = cls.load_dictionary(os.path.join(args.data[0], 'dict.{}.txt'.format(args.target_lang)))
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        print('| [{}] dictionary: {} types'.format(args.source_lang, len(src_dict)))
        print('| [{}] dictionary: {} types'.format(args.target_lang, len(tgt_dict)))

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def split_exists(split, src, tgt, lang, data_path):
            filename = os.path.join(data_path, '{}.{}-{}.{}'.format(split, src, tgt, lang))
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

        src_datasets = []
        tgt_datasets = []

        data_paths = self.args.data

        for dk, data_path in enumerate(data_paths):
            for k in itertools.count():
                split_k = split + (str(k) if k > 0 else '')

                # infer langcode
                src, tgt = self.args.source_lang, self.args.target_lang
                if split_exists(split_k, src, tgt, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, src, tgt))
                elif split_exists(split_k, tgt, src, src, data_path):
                    prefix = os.path.join(data_path, '{}.{}-{}.'.format(split_k, tgt, src))
                else:
                    if k > 0 or dk > 0:
                        break
                    else:
                        raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

                src_datasets.append(indexed_dataset(prefix + src, self.src_dict))
                tgt_datasets.append(indexed_dataset(prefix + tgt, self.tgt_dict))

                print('| {} {} {} examples'.format(data_path, split_k, len(src_datasets[-1])))

                if not combine:
                    break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = self.args.upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        self.datasets[split] = LanguagePairDataset(
            src_dataset, src_dataset.sizes, self.src_dict,
            tgt_dataset, tgt_dataset.sizes, self.tgt_dict,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        return LanguagePairDataset(src_tokens, src_lengths, self.source_dictionary)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary`."""
        return self.src_dict

    @property
    def target_dictionary(self):
        """Return the target :class:`~fairseq.data.Dictionary`."""
        return self.tgt_dict

