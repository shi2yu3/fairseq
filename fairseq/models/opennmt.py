# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LayerNorm,
    LearnedPositionalEmbedding, MultiheadAttention, SinusoidalPositionalEmbedding,
)

from fairseq.models.sru import CheckSRU

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel,
    FairseqModel, register_model, register_model_architecture,
)


@register_model('opennmt')
class OpenNMTModel(FairseqModel):
    """
    from https://github.com/OpenNMT/OpenNMT-py

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser.
        These options are passed to the construction of the model.
        Be careful with these as they will be used during translation.
        """
        # fmt: off
        # Embedding Options
        parser.add_argument('--src_word_vec_size', '-src_word_vec_size',
                            type=int, default=500,
                            help='Word embedding size for src.')
        parser.add_argument('--tgt_word_vec_size', '-tgt_word_vec_size',
                  type=int, default=500,
                  help='Word embedding size for tgt.')
        parser.add_argument('--word_vec_size', '-word_vec_size', type=int, default=-1,
                  help='Word embedding size for src and tgt.')

        parser.add_argument('--share_decoder_embeddings', '-share_decoder_embeddings',
                  action='store_true',
                  help="Use a shared weight matrix for the input and "
                       "output word  embeddings in the decoder.")
        parser.add_argument('--share_embeddings', '-share_embeddings', action='store_true',
                  help="Share the word embeddings between encoder "
                       "and decoder. Need to use shared dictionary for this "
                       "option.")
        parser.add_argument('--position_encoding', '-position_encoding', action='store_true',
                  help="Use a sin to mark relative words positions. "
                       "Necessary for non-RNN style models.")

        group = parser.add_argument_group('Model-Embedding Features')
        parser.add_argument('--feat_merge', '-feat_merge', type=str, default='concat',
                  choices=['concat', 'sum', 'mlp'],
                  help="Merge action for incorporating features embeddings. "
                       "Options [concat|sum|mlp].")
        parser.add_argument('--feat_vec_size', '-feat_vec_size', type=int, default=-1,
                  help="If specified, feature embedding sizes "
                       "will be set to this. Otherwise, feat_vec_exponent "
                       "will be used.")
        parser.add_argument('--feat_vec_exponent', '-feat_vec_exponent',
                  type=float, default=0.7,
                  help="If -feat_merge_size is not set, feature "
                       "embedding sizes will be set to N^feat_vec_exponent "
                       "where N is the number of values the feature takes.")

        # Encoder-Decoder Options
        group = parser.add_argument_group('Model- Encoder-Decoder')
        parser.add_argument('--model_type', '-model_type', default='text',
                  choices=['text', 'img', 'audio'],
                  help="Type of source model to use. Allows "
                       "the system to incorporate non-text inputs. "
                       "Options are [text|img|audio].")
        parser.add_argument('--model_dtype', '-model_dtype', default='fp32',
                  choices=['fp32', 'fp16'],
                  help='Data type of the model.')

        parser.add_argument('--encoder_type', '-encoder_type', type=str, default='rnn',
                  choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
                  help="Type of encoder layer to use. Non-RNN layers "
                       "are experimental. Options are "
                       "[rnn|brnn|mean|transformer|cnn].")
        parser.add_argument('--decoder_type', '-decoder_type', type=str, default='rnn',
                  choices=['rnn', 'transformer', 'cnn'],
                  help="Type of decoder layer to use. Non-RNN layers "
                       "are experimental. Options are "
                       "[rnn|transformer|cnn].")

        parser.add_argument('--layers', '-layers', type=int, default=-1,
                  help='Number of layers in enc/dec.')
        parser.add_argument('--enc_layers', '-enc_layers', type=int, default=2,
                  help='Number of layers in the encoder')
        parser.add_argument('--dec_layers', '-dec_layers', type=int, default=2,
                  help='Number of layers in the decoder')
        parser.add_argument('--rnn_size', '-rnn_size', type=int, default=-1,
                  help="Size of rnn hidden states. Overwrites "
                       "enc_rnn_size and dec_rnn_size")
        parser.add_argument('--enc_rnn_size', '-enc_rnn_size', type=int, default=500,
                  help="Size of encoder rnn hidden states. "
                       "Must be equal to dec_rnn_size except for "
                       "speech-to-text.")
        parser.add_argument('--dec_rnn_size', '-dec_rnn_size', type=int, default=500,
                  help="Size of decoder rnn hidden states. "
                       "Must be equal to enc_rnn_size except for "
                       "speech-to-text.")
        parser.add_argument('--audio_enc_pooling', '-audio_enc_pooling',
                  type=str, default='1',
                  help="The amount of pooling of audio encoder, "
                       "either the same amount of pooling across all layers "
                       "indicated by a single number, or different amounts of "
                       "pooling per layer separated by comma.")
        parser.add_argument('--cnn_kernel_width', '-cnn_kernel_width', type=int, default=3,
                  help="Size of windows in the cnn, the kernel_size is "
                       "(cnn_kernel_width, 1) in conv layer")

        parser.add_argument('--input_feed', '-input_feed', type=int, default=1,
                  help="Feed the context vector at each time step as "
                       "additional input (via concatenation with the word "
                       "embeddings) to the decoder.")
        parser.add_argument('--bridge', '-bridge', action="store_true",
                  help="Have an additional layer between the last encoder "
                       "state and the first decoder state")
        parser.add_argument('--rnn_type', '-rnn_type', type=str, default='LSTM',
                  choices=['LSTM', 'GRU', 'SRU'],
                  action=CheckSRU,
                  help="The gate type to use in the RNNs")

        parser.add_argument('--context_gate', '-context_gate', type=str, default=None,
                  choices=['source', 'target', 'both'],
                  help="Type of context gate to use. "
                       "Do not select for no context gate.")

        # Attention options
        parser.add_argument('--global_attention', '-global_attention',
                  type=str, default='general',
                  choices=['dot', 'general', 'mlp', 'none'],
                  help="The attention type to use: "
                       "dotprod or general (Luong) or MLP (Bahdanau)")
        parser.add_argument('--global_attention_function', '-global_attention_function',
                  type=str, default="softmax", choices=["softmax", "sparsemax"])
        parser.add_argument('--self_attn_type', '-self_attn_type',
                  type=str, default="scaled-dot",
                  help='Self attention type in Transformer decoder '
                       'layer -- currently "scaled-dot" or "average" ')
        parser.add_argument('--max_relative_positions', '-max_relative_positions',
                  type=int, default=0,
                  help="Maximum distance between inputs in relative "
                       "positions representations. "
                       "For more detailed information, see: "
                       "https://arxiv.org/pdf/1803.02155.pdf")
        parser.add_argument('--heads', '-heads', type=int, default=8,
                  help='Number of heads for transformer self-attention')
        parser.add_argument('--transformer_ff', '-transformer_ff', type=int, default=2048,
                  help='Size of hidden transformer feed-forward')

        # Generator and loss options.
        parser.add_argument('--copy_attn', '-copy_attn', action="store_true",
                  help='Train copy attention layer.')
        parser.add_argument('--copy_attn_type', '-copy_attn_type',
                  type=str, default=None,
                  choices=['dot', 'general', 'mlp', 'none'],
                  help="The copy attention type to use. Leave as None to use "
                       "the same as -global_attention.")
        parser.add_argument('--generator_function', '-generator_function', default="softmax",
                  choices=["softmax", "sparsemax"],
                  help="Which function to use for generating "
                       "probabilities over the target vocabulary (choices: "
                       "softmax, sparsemax)")
        parser.add_argument('--copy_attn_force', '-copy_attn_force', action="store_true",
                  help='When available, train to copy.')
        parser.add_argument('--reuse_copy_attn', '-reuse_copy_attn', action="store_true",
                  help="Reuse standard attention for copy")
        parser.add_argument('--copy_loss_by_seqlength', '-copy_loss_by_seqlength',
                  action="store_true",
                  help="Divide copy loss by length of sequence")
        parser.add_argument('--coverage_attn', '-coverage_attn', action="store_true",
                  help='Train a coverage attention layer.')
        parser.add_argument('--lambda_coverage', '-lambda_coverage', type=float, default=1,
                  help='Lambda value for coverage.')
        parser.add_argument('--loss_scale', '-loss_scale', type=float, default=0,
                  help="For FP16 training, the static loss scale to use. If not "
                       "set, the loss scale is dynamically computed.")
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        opennmt(args)




        if not hasattr(args, 'max_source_positions'):
            args.max_source_positions = 1024
        if not hasattr(args, 'max_target_positions'):
            args.max_target_positions = 1024

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModel(encoder, decoder)


    @classmethod
    def update_argss(cls, args):
        if args.word_vec_size > 0:
            args.src_word_vec_size = args.word_vec_size
            args.tgt_word_vec_size = args.word_vec_size

        if args.layers > 0:
            args.enc_layers = args.layers
            args.dec_layers = args.layers

        if args.rnn_size > 0:
            args.enc_rnn_size = args.rnn_size
            args.dec_rnn_size = args.rnn_size

        if args.copy_attn_type is None:
            args.copy_attn_type = args.global_attention


    @classmethod
    def validate_argss(cls, args):
        assert args.model_type in ["text", "img", "audio"], \
            "Unsupported model type %s" % args.model_type

        # this check is here because audio allows the encoder and decoder to
        # be different sizes, but other model types do not yet
        same_size = args.enc_rnn_size == args.dec_rnn_size
        assert args.model_type == 'audio' or same_size, \
            "The encoder and decoder rnns must be the same size for now"

        assert args.rnn_type != "SRU" or args.gpu_ranks, \
            "Using SRU requires -gpu_ranks set."
        if args.share_embeddings:
            if args.model_type != "text":
                raise AssertionError(
                    "--share_embeddings requires --model_type text.")
        # if args.model_dtype == "fp16":
        #     logger.warning(
        #         "FP16 is experimental, the generated checkpoints may "
        #         "be incompatible with a future version")


@register_model_architecture('opennmt', 'opennmt')
def opennmt(args):
    # Embedding Options
    args.src_word_vec_size = getattr(args, 'src_word_vec_size', 500)
    args.tgt_word_vec_size = getattr(args, 'tgt_word_vec_size', 500)
    args.word_vec_size = getattr(args, 'word_vec_size', -1)

    args.share_decoder_embeddings = getattr(args, 'share_decoder_embeddings', False)
    args.share_embeddings = getattr(args, 'share_embeddings', False)
    args.position_encoding = getattr(args, 'position_encoding', False)

    args.feat_merge = getattr(args, 'feat_merge', 'concat')
    args.feat_vec_size = getattr(args, 'feat_vec_size', -1)
    args.feat_vec_exponent = getattr(args, 'feat_vec_exponent', 0.7)

    # Encoder-Decoder Options
    args.model_type = getattr(args, 'model_type', 'text')
    args.model_dtype = getattr(args, 'model_dtype', 'fp32')

    args.encoder_type = getattr(args, 'encoder_type', 'rnn')
    args.decoder_type = getattr(args, 'decoder_type', 'rnn')

    args.layers = getattr(args, 'layers', -1)
    args.enc_layers = getattr(args, 'enc_layers', 2)
    args.dec_layers = getattr(args, 'dec_layers', 2)
    args.rnn_size = getattr(args, 'rnn_size', -1)
    args.enc_rnn_size = getattr(args, 'enc_rnn_size', 500)
    args.dec_rnn_size = getattr(args, 'dec_rnn_size', 500)
    args.audio_enc_pooling = getattr(args, 'audio_enc_pooling', '1')
    args.cnn_kernel_width = getattr(args, 'cnn_kernel_width', 3)

    args.input_feed = getattr(args, 'input_feed', 1)
    args.bridge = getattr(args, 'bridge', False)
    args.rnn_type = getattr(args, 'rnn_type', 'LSTM')
    # args.residual = getattr(args, 'residual', False)

    args.context_gate = getattr(args, 'context_gate', None)

    # Attention options
    args.global_attention = getattr(args, 'global_attention', 'general')
    args.global_attention_function = getattr(args, 'global_attention_function', "softmax")
    args.self_attn_type = getattr(args, 'self_attn_type', "scaled-dot")
    args.max_relative_positions = getattr(args, 'max_relative_positions', 0)
    args.heads = getattr(args, 'heads', 8)
    args.transformer_ff = getattr(args, 'transformer_ff', 2048)

    # Generator and loss options.
    args.copy_attn = getattr(args, 'copy_attn', False)
    args.copy_attn_type = getattr(args, 'copy_attn_type', None)
    args.generator_function = getattr(args, 'generator_function', "softmax")
    args.copy_attn_force = getattr(args, 'copy_attn_force', False)
    args.reuse_copy_attn = getattr(args, 'reuse_copy_attn', False)
    args.copy_loss_by_seqlength = getattr(args, 'copy_loss_by_seqlength', False)
    args.coverage_attn = getattr(args, 'coverage_attn', False)
    args.lambda_coverage = getattr(args, 'lambda_coverage', 1)
    args.loss_scale = getattr(args, 'loss_scale', 0)


@register_model_architecture('opennmt', 'opennmt_cnndm')
def opennmt_cnndm(args):
    '''
                -train_steps 200000 \
                -max_grad_norm 2 \
                -dropout 0. \
                -batch_size 16 \
                -valid_batch_size 16 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -copy_loss_by_seqlength \
                -seed 777 \
                -world_size 2 \
                -gpu_ranks 0 1
    '''
    args.word_vec_size = getattr(args, 'word_vec_size', 128)
    args.encoder_type = getattr(args, 'encoder_type', 'brnn')
    args.layers = getattr(args, 'layers', 1)
    args.rnn_size = getattr(args, 'rnn_size', 512)
    args.bridge = getattr(args, 'bridge', True)
    args.global_attention = getattr(args, 'global_attention', 'mlp')
    args.copy_attn = getattr(args, 'copy_attn', True)
    args.reuse_copy_attn = getattr(args, 'reuse_copy_attn', True)


@register_model_architecture('opennmt', 'opennmt_gigaword')
def opennmt_gigaword(args):
    '''
                   -max_grad_norm 0 \
                   -optim adam \
                   -dropout 0\.2 \
                   -param_init 0 \
                   -warmup_steps 8000 \
                   -learning_rate 2 \
                   -decay_method noam \
                   -label_smoothing 0.1 \
                   -adam_beta2 0.998 \
                   -batch_size 4096 \
                   -batch_type tokens \
                   -normalization tokens \
                   -max_generator_batches 2 \
                   -train_steps 200000 \
                   -accum_count 4 \
                   -param_init_glorot \
                   -world_size 2 \
                   -gpu_ranks 0 1
    '''
    args.word_vec_size = getattr(args, 'word_vec_size', 512)
    args.share_embeddings = getattr(args, 'share_embeddings', True)
    args.position_encoding = getattr(args, 'position_encoding', True)
    args.encoder_type = getattr(args, 'encoder_type', 'transformer')
    args.decoder_type = getattr(args, 'decoder_type', 'transformer')
    args.layers = getattr(args, 'layers', 4)
    args.rnn_size = getattr(args, 'rnn_size', 512)
    args.copy_attn = getattr(args, 'copy_attn', True)

