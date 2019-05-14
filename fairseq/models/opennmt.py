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
from torch.nn.init import xavier_uniform_

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LayerNorm,
    LearnedPositionalEmbedding, MultiheadAttention, SinusoidalPositionalEmbedding,
)

from fairseq.modules.onmt import (
    sparse_activations, util_class, copy_generator, Embeddings
)

from .onmt.model_builder import build_embeddings, build_encoder, build_decoder
from .onmt.models import NMTModel
from .onmt.models.sru import CheckSRU

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel,
    FairseqModel, register_model, register_model_architecture,
)


@register_model('opennmt')
class OpenNMTModel(FairseqModel):
    """
    from https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """
        These options are passed to the construction of the model.
        Be careful with these as they will be used during translation.
        """

        # fmt: off

        # Optimization Options
        parser.add_argument('--dropout', '-dropout', type=float, default=0.3,
                            help="Dropout probability; applied in LSTM stacks.")

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
        # Pretrained word vectors
        parser.add_argument('--pre_word_vecs_enc', '-pre_word_vecs_enc',
                  help="If a valid path is specified, then this will load "
                       "pretrained word embeddings on the encoder side. "
                       "See README for specific formatting instructions.")
        parser.add_argument('--pre_word_vecs_dec', '-pre_word_vecs_dec',
                  help="If a valid path is specified, then this will load "
                       "pretrained word embeddings on the decoder side. "
                       "See README for specific formatting instructions.")
        # Fixed word vectors
        parser.add_argument('--fix_word_vecs_enc', '-fix_word_vecs_enc',
                  action='store_true',
                  help="Fix word embeddings on the encoder side.")
        parser.add_argument('--fix_word_vecs_dec', '-fix_word_vecs_dec',
                  action='store_true',
                  help="Fix word embeddings on the decoder side.")

        # Encoder-Decoder Options
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
        base_architecture(args)

        # Build embeddings.
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

        if args.share_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.src_word_vec_size != args.tgt_word_vec_size:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.pre_word_vecs_dec and (
                    args.pre_word_vecs_dec != args.pre_word_vecs_enc):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.src_word_vec_size, args.pre_word_vecs_enc
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.src_word_vec_size, args.pre_word_vecs_enc
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.tgt_word_vec_size, args.pre_word_vecs_dec
            )

        # convert to OpenNMT Embeddings class
        src_emb = Embeddings(
            word_vec_size=args.src_word_vec_size,
            position_encoding=args.position_encoding,
            feat_merge=args.feat_merge,
            feat_vec_exponent=args.feat_vec_exponent,
            feat_vec_size=args.feat_vec_size,
            dropout=args.dropout,
            word_padding_idx=src_dict.pad(),
            feat_padding_idx=[],
            word_vocab_size=len(src_dict),
            feat_vocab_sizes=[],
            sparse=args.optimizer == "sparseadam",
            fix_word_vecs=args.fix_word_vecs_enc
        )
        tgt_emb = Embeddings(
            word_vec_size=args.tgt_word_vec_size,
            position_encoding=args.position_encoding,
            feat_merge=args.feat_merge,
            feat_vec_exponent=args.feat_vec_exponent,
            feat_vec_size=args.feat_vec_size,
            dropout=args.dropout,
            word_padding_idx=tgt_dict.pad(),
            feat_padding_idx=[],
            word_vocab_size=len(tgt_dict),
            feat_vocab_sizes=[],
            sparse=args.optimizer == "sparseadam",
            fix_word_vecs=args.fix_word_vecs_dec
        )

        # Build encoder.
        encoder = build_encoder(args, src_emb)

        # Build decoder.
        decoder = build_decoder(args, tgt_emb)

        # Build NMTModel(= encoder + decoder).
        model = NMTModel(encoder, decoder)

        # Build Generator.
        if not args.copy_attn:
            if args.generator_function == "sparsemax":
                gen_func = sparse_activations.LogSparsemax(dim=-1)
            else:
                gen_func = nn.LogSoftmax(dim=-1)
            generator = nn.Sequential(
                nn.Linear(args.dec_rnn_size,
                          len(task.fields["tgt"].base_field.vocab)),
                util_class.Cast(torch.float32),
                gen_func
            )
            if args.share_decoder_embeddings:
                generator[0].weight = decoder.embeddings.word_lut.weight
        else:
            vocab_size = len(tgt_dict)
            pad_idx = tgt_dict.pad()
            generator = copy_generator.CopyGenerator(args.dec_rnn_size, vocab_size, pad_idx)

        # Load the model states from checkpoint or initialize them.
        if False: # checkpoint is not None:
            # This preserves backward-compat for models using customed layernorm
            def fix_key(s):
                s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.b_2',
                           r'\1.layer_norm\2.bias', s)
                s = re.sub(r'(.*)\.layer_norm((_\d+)?)\.a_2',
                           r'\1.layer_norm\2.weight', s)
                return s

            checkpoint['model'] = {fix_key(k): v
                                   for k, v in checkpoint['model'].items()}
            # end of patch for backward compatibility

            model.load_state_dict(checkpoint['model'], strict=False)
            generator.load_state_dict(checkpoint['generator'], strict=False)
        else:
            if args.param_init != 0.0:
                for p in model.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
                for p in generator.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in model.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
                for p in generator.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)

            if hasattr(model.encoder, 'embeddings'):
                model.encoder.embeddings.load_pretrained_vectors(
                    args.pre_word_vecs_enc)
            if hasattr(model.decoder, 'embeddings'):
                model.decoder.embeddings.load_pretrained_vectors(
                    args.pre_word_vecs_dec)

        model.generator = generator
        return model


# def config_opts(parser):
#     parser.add_argument('--config', '-config', required=False,
#                         is_config_file_arg=True, help='config file path')
#     parser.add_argument('--save_config', '-save_config', required=False,
#                         is_write_out_config_file_arg=True,
#                         help='config file save path')


# def model_opts(parser):
#     """
#     These options are passed to the construction of the model.
#     Be careful with these as they will be used during translation.
#     """
#
#     # Embedding Options
#     parser.add_argument('--src_word_vec_size', '-src_word_vec_size',
#                         type=int, default=500,
#                         help='Word embedding size for src.')
#     parser.add_argument('--tgt_word_vec_size', '-tgt_word_vec_size',
#                         type=int, default=500,
#                         help='Word embedding size for tgt.')
#     parser.add_argument('--word_vec_size', '-word_vec_size', type=int, default=-1,
#                         help='Word embedding size for src and tgt.')
#
#     parser.add_argument('--share_decoder_embeddings', '-share_decoder_embeddings',
#                         action='store_true',
#                         help="Use a shared weight matrix for the input and "
#                              "output word  embeddings in the decoder.")
#     parser.add_argument('--share_embeddings', '-share_embeddings', action='store_true',
#                         help="Share the word embeddings between encoder "
#                              "and decoder. Need to use shared dictionary for this "
#                              "option.")
#     parser.add_argument('--position_encoding', '-position_encoding', action='store_true',
#                         help="Use a sin to mark relative words positions. "
#                              "Necessary for non-RNN style models.")
#
#     parser.add_argument('--feat_merge', '-feat_merge', type=str, default='concat',
#                         choices=['concat', 'sum', 'mlp'],
#                         help="Merge action for incorporating features embeddings. "
#                              "Options [concat|sum|mlp].")
#     parser.add_argument('--feat_vec_size', '-feat_vec_size', type=int, default=-1,
#                         help="If specified, feature embedding sizes "
#                              "will be set to this. Otherwise, feat_vec_exponent "
#                              "will be used.")
#     parser.add_argument('--feat_vec_exponent', '-feat_vec_exponent',
#                         type=float, default=0.7,
#                         help="If -feat_merge_size is not set, feature "
#                              "embedding sizes will be set to N^feat_vec_exponent "
#                              "where N is the number of values the feature takes.")
#
#     # Encoder-Decoder Options
#     parser.add_argument('--model_type', '-model_type', default='text',
#                         choices=['text', 'img', 'audio'],
#                         help="Type of source model to use. Allows "
#                              "the system to incorporate non-text inputs. "
#                              "Options are [text|img|audio].")
#     parser.add_argument('--model_dtype', '-model_dtype', default='fp32',
#                         choices=['fp32', 'fp16'],
#                         help='Data type of the model.')
#
#     parser.add_argument('--encoder_type', '-encoder_type', type=str, default='rnn',
#                         choices=['rnn', 'brnn', 'mean', 'transformer', 'cnn'],
#                         help="Type of encoder layer to use. Non-RNN layers "
#                              "are experimental. Options are "
#                              "[rnn|brnn|mean|transformer|cnn].")
#     parser.add_argument('--decoder_type', '-decoder_type', type=str, default='rnn',
#                         choices=['rnn', 'transformer', 'cnn'],
#                         help="Type of decoder layer to use. Non-RNN layers "
#                              "are experimental. Options are "
#                              "[rnn|transformer|cnn].")
#
#     parser.add_argument('--layers', '-layers', type=int, default=-1,
#                         help='Number of layers in enc/dec.')
#     parser.add_argument('--enc_layers', '-enc_layers', type=int, default=2,
#                         help='Number of layers in the encoder')
#     parser.add_argument('--dec_layers', '-dec_layers', type=int, default=2,
#                         help='Number of layers in the decoder')
#     parser.add_argument('--rnn_size', '-rnn_size', type=int, default=-1,
#                         help="Size of rnn hidden states. Overwrites "
#                              "enc_rnn_size and dec_rnn_size")
#     parser.add_argument('--enc_rnn_size', '-enc_rnn_size', type=int, default=500,
#                         help="Size of encoder rnn hidden states. "
#                              "Must be equal to dec_rnn_size except for "
#                              "speech-to-text.")
#     parser.add_argument('--dec_rnn_size', '-dec_rnn_size', type=int, default=500,
#                         help="Size of decoder rnn hidden states. "
#                              "Must be equal to enc_rnn_size except for "
#                              "speech-to-text.")
#     parser.add_argument('--audio_enc_pooling', '-audio_enc_pooling',
#                         type=str, default='1',
#                         help="The amount of pooling of audio encoder, "
#                              "either the same amount of pooling across all layers "
#                              "indicated by a single number, or different amounts of "
#                              "pooling per layer separated by comma.")
#     parser.add_argument('--cnn_kernel_width', '-cnn_kernel_width', type=int, default=3,
#                         help="Size of windows in the cnn, the kernel_size is "
#                              "(cnn_kernel_width, 1) in conv layer")
#
#     parser.add_argument('--input_feed', '-input_feed', type=int, default=1,
#                         help="Feed the context vector at each time step as "
#                              "additional input (via concatenation with the word "
#                              "embeddings) to the decoder.")
#     parser.add_argument('--bridge', '-bridge', action="store_true",
#                         help="Have an additional layer between the last encoder "
#                              "state and the first decoder state")
#     parser.add_argument('--rnn_type', '-rnn_type', type=str, default='LSTM',
#                         choices=['LSTM', 'GRU', 'SRU'],
#                         action=CheckSRU,
#                         help="The gate type to use in the RNNs")
#
#     parser.add_argument('--context_gate', '-context_gate', type=str, default=None,
#                         choices=['source', 'target', 'both'],
#                         help="Type of context gate to use. "
#                              "Do not select for no context gate.")
#
#     # Attention options
#     parser.add_argument('--global_attention', '-global_attention',
#                         type=str, default='general',
#                         choices=['dot', 'general', 'mlp', 'none'],
#                         help="The attention type to use: "
#                              "dotprod or general (Luong) or MLP (Bahdanau)")
#     parser.add_argument('--global_attention_function', '-global_attention_function',
#                         type=str, default="softmax", choices=["softmax", "sparsemax"])
#     parser.add_argument('--self_attn_type', '-self_attn_type',
#                         type=str, default="scaled-dot",
#                         help='Self attention type in Transformer decoder '
#                              'layer -- currently "scaled-dot" or "average" ')
#     parser.add_argument('--max_relative_positions', '-max_relative_positions',
#                         type=int, default=0,
#                         help="Maximum distance between inputs in relative "
#                              "positions representations. "
#                              "For more detailed information, see: "
#                              "https://arxiv.org/pdf/1803.02155.pdf")
#     parser.add_argument('--heads', '-heads', type=int, default=8,
#                         help='Number of heads for transformer self-attention')
#     parser.add_argument('--transformer_ff', '-transformer_ff', type=int, default=2048,
#                         help='Size of hidden transformer feed-forward')
#
#     # Generator and loss options.
#     parser.add_argument('--copy_attn', '-copy_attn', action="store_true",
#                         help='Train copy attention layer.')
#     parser.add_argument('--copy_attn_type', '-copy_attn_type',
#                         type=str, default=None,
#                         choices=['dot', 'general', 'mlp', 'none'],
#                         help="The copy attention type to use. Leave as None to use "
#                              "the same as -global_attention.")
#     parser.add_argument('--generator_function', '-generator_function', default="softmax",
#                         choices=["softmax", "sparsemax"],
#                         help="Which function to use for generating "
#                              "probabilities over the target vocabulary (choices: "
#                              "softmax, sparsemax)")
#     parser.add_argument('--copy_attn_force', '-copy_attn_force', action="store_true",
#                         help='When available, train to copy.')
#     parser.add_argument('--reuse_copy_attn', '-reuse_copy_attn', action="store_true",
#                         help="Reuse standard attention for copy")
#     parser.add_argument('--copy_loss_by_seqlength', '-copy_loss_by_seqlength',
#                         action="store_true",
#                         help="Divide copy loss by length of sequence")
#     parser.add_argument('--coverage_attn', '-coverage_attn', action="store_true",
#                         help='Train a coverage attention layer.')
#     parser.add_argument('--lambda_coverage', '-lambda_coverage', type=float, default=1,
#                         help='Lambda value for coverage.')
#     parser.add_argument('--loss_scale', '-loss_scale', type=float, default=0,
#                         help="For FP16 training, the static loss scale to use. If not "
#                              "set, the loss scale is dynamically computed.")


# def train_opts(parser):
#     """ Training and saving options """
#
#     # Init options
#     parser.add_argument('--param_init', '-param_init', type=float, default=0.1,
#               help="Parameters are initialized over uniform distribution "
#                    "with support (-param_init, param_init). "
#                    "Use 0 to not use initialization")
#     parser.add_argument('--param_init_glorot', '-param_init_glorot', action='store_true',
#               help="Init parameters with xavier_uniform. "
#                    "Required for transformer.")
#
#     parser.add_argument('--train_from', '-train_from', default='', type=str,
#               help="If training from a checkpoint then this is the "
#                    "path to the pretrained model's state_dict.")
#     parser.add_argument('--reset_optim', '-reset_optim', default='none',
#               choices=['none', 'all', 'states', 'keep_states'],
#               help="Optimization resetter when train_from.")
#
#     # Pretrained word vectors
#     parser.add_argument('--pre_word_vecs_enc', '-pre_word_vecs_enc',
#               help="If a valid path is specified, then this will load "
#                    "pretrained word embeddings on the encoder side. "
#                    "See README for specific formatting instructions.")
#     parser.add_argument('--pre_word_vecs_dec', '-pre_word_vecs_dec',
#               help="If a valid path is specified, then this will load "
#                    "pretrained word embeddings on the decoder side. "
#                    "See README for specific formatting instructions.")
#     # Fixed word vectors
#     parser.add_argument('--fix_word_vecs_enc', '-fix_word_vecs_enc',
#               action='store_true',
#               help="Fix word embeddings on the encoder side.")
#     parser.add_argument('--fix_word_vecs_dec', '-fix_word_vecs_dec',
#               action='store_true',
#               help="Fix word embeddings on the decoder side.")
#
#     # Optimization options
#     # parser.add_argument('--batch_size', '-batch_size', type=int, default=64,
#     #           help='Maximum batch size for training')
#     # parser.add_argument('--batch_type', '-batch_type', default='sents',
#     #           choices=["sents", "tokens"],
#     #           help="Batch grouping for batch_size. Standard "
#     #                "is sents. Tokens will do dynamic batching")
#     # parser.add_argument('--normalization', '-normalization', default='sents',
#     #           choices=["sents", "tokens"],
#     #           help='Normalization method of the gradient.')
#     # parser.add_argument('--accum_count', '-accum_count', type=int, nargs='+',
#     #           default=[1],
#     #           help="Accumulate gradient this many times. "
#     #                "Approximately equivalent to updating "
#     #                "batch_size * accum_count batches at once. "
#     #                "Recommended for Transformer.")
#     # parser.add_argument('--accum_steps', '-accum_steps', type=int, nargs='+',
#     #           default=[0], help="Steps at which accum_count values change")
#     # parser.add_argument('--valid_steps', '-valid_steps', type=int, default=10000,
#     #           help='Perfom validation every X steps')
#     # parser.add_argument('--valid_batch_size', '-valid_batch_size', type=int, default=32,
#     #           help='Maximum batch size for validation')
#     # parser.add_argument('--max_generator_batches', '-max_generator_batches',
#     #           type=int, default=32,
#     #           help="Maximum batches of words in a sequence to run "
#     #                "the generator on in parallel. Higher is faster, but "
#     #                "uses more memory. Set to 0 to disable.")
#     # parser.add_argument('--train_steps', '-train_steps', type=int, default=100000,
#     #           help='Number of training steps')
#     # parser.add_argument('--single_pass', '-single_pass', action='store_true',
#     #           help="Make a single pass over the training dataset.")
#     # parser.add_argument('--epochs', '-epochs', type=int, default=0,
#     #           help='Deprecated epochs see train_steps')
#     # parser.add_argument('--early_stopping', '-early_stopping', type=int, default=0,
#     #           help='Number of validation steps without improving.')
#     # parser.add_argument('--early_stopping_criteria', '-early_stopping_criteria',
#     #           nargs="*", default=None,
#     #           help='Criteria to use for early stopping.')
#     # parser.add_argument('--optim', '-optim', default='sgd',
#     #           choices=['sgd', 'adagrad', 'adadelta', 'adam',
#     #                    'sparseadam', 'adafactor', 'fusedadam'],
#     #           help="Optimization method.")
#     # parser.add_argument('--adagrad_accumulator_init', '-adagrad_accumulator_init',
#     #           type=float, default=0,
#     #           help="Initializes the accumulator values in adagrad. "
#     #                "Mirrors the initial_accumulator_value option "
#     #                "in the tensorflow adagrad (use 0.1 for their default).")
#     # parser.add_argument('--max_grad_norm', '-max_grad_norm', type=float, default=5,
#     #           help="If the norm of the gradient vector exceeds this, "
#     #                "renormalize it to have the norm equal to "
#     #                "max_grad_norm")
#     parser.add_argument('--dropout', '-dropout', type=float, default=0.3,
#                         help="Dropout probability; applied in LSTM stacks.")
#     parser.add_argument('--truncated_decoder', '-truncated_decoder', type=int, default=0,
#               help="""Truncated bptt.""")
#     # parser.add_argument('--adam_beta1', '-adam_beta1', type=float, default=0.9,
#     #           help="The beta1 parameter used by Adam. "
#     #                "Almost without exception a value of 0.9 is used in "
#     #                "the literature, seemingly giving good results, "
#     #                "so we would discourage changing this value from "
#     #                "the default without due consideration.")
#     # parser.add_argument('--adam_beta2', '-adam_beta2', type=float, default=0.999,
#     #           help='The beta2 parameter used by Adam. '
#     #                'Typically a value of 0.999 is recommended, as this is '
#     #                'the value suggested by the original paper describing '
#     #                'Adam, and is also the value adopted in other frameworks '
#     #                'such as Tensorflow and Kerras, i.e. see: '
#     #                'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
#     #                'Optimizer or '
#     #                'https://keras.io/optimizers/ . '
#     #                'Whereas recently the paper "Attention is All You Need" '
#     #                'suggested a value of 0.98 for beta2, this parameter may '
#     #                'not work well for normal models / default '
#     #                'baselines.')
#     # parser.add_argument('--label_smoothing', '-label_smoothing', type=float, default=0.0,
#     #           help="Label smoothing value epsilon. "
#     #                "Probabilities of all non-true labels "
#     #                "will be smoothed by epsilon / (vocab_size - 1). "
#     #                "Set to zero to turn off label smoothing. "
#     #                "For more detailed information, see: "
#     #                "https://arxiv.org/abs/1512.00567")
#     # parser.add_argument('--average_decay', '-average_decay', type=float, default=0,
#     #           help="Moving average decay. "
#     #                "Set to other than 0 (e.g. 1e-4) to activate. "
#     #                "Similar to Marian NMT implementation: "
#     #                "http://www.aclweb.org/anthology/P18-4020 "
#     #                "For more detail on Exponential Moving Average: "
#     #                "https://en.wikipedia.org/wiki/Moving_average")
#     # parser.add_argument('--average_every', '-average_every', type=int, default=1,
#     #           help="Step for moving average. "
#     #                "Default is every update, "
#     #                "if -average_decay is set.")
#
#     # learning rate
#     group = parser.add_argument_group('Optimization- Rate')
#     parser.add_argument('--learning_rate', '-learning_rate', type=float, default=1.0,
#               help="Starting learning rate. "
#                    "Recommended settings: sgd = 1, adagrad = 0.1, "
#                    "adadelta = 1, adam = 0.001")
#     parser.add_argument('--learning_rate_decay', '-learning_rate_decay',
#               type=float, default=0.5,
#               help="If update_learning_rate, decay learning rate by "
#                    "this much if steps have gone past "
#                    "start_decay_steps")
#     parser.add_argument('--start_decay_steps', '-start_decay_steps',
#               type=int, default=50000,
#               help="Start decaying every decay_steps after "
#                    "start_decay_steps")
#     parser.add_argument('--decay_steps', '-decay_steps', type=int, default=10000,
#               help="Decay every decay_steps")
#
#     parser.add_argument('--decay_method', '-decay_method', type=str, default="none",
#               choices=['noam', 'noamwd', 'rsqrt', 'none'],
#               help="Use a custom decay rate.")
#     parser.add_argument('--warmup_steps', '-warmup_steps', type=int, default=4000,
#               help="Number of warmup steps for custom decay.")
#
#     group = parser.add_argument_group('Logging')
#     parser.add_argument('--report_every', '-report_every', type=int, default=50,
#               help="Print stats at this interval.")
#     parser.add_argument('--log_file', '-log_file', type=str, default="",
#               help="Output logs to a file under this path.")
#     parser.add_argument('--log_file_level', '-log_file_level', type=str,
#               action=StoreLoggingLevelAction,
#               choices=StoreLoggingLevelAction.CHOICES,
#               default="0")
#     parser.add_argument('--exp_host', '-exp_host', type=str, default="",
#               help="Send logs to this crayon server.")
#     parser.add_argument('--exp', '-exp', type=str, default="",
#               help="Name of the experiment for logging.")
#     # Use TensorboardX for visualization during training
#     parser.add_argument('--tensorboard', '-tensorboard', action="store_true",
#               help="Use tensorboardX for visualization during training. "
#                    "Must have the library tensorboardX.")
#     parser.add_argument("--tensorboard_log_dir", "-tensorboard_log_dir",
#               type=str, default="runs/onmt",
#               help="Log directory for Tensorboard. "
#                    "This is also the name of the run.")
#
#     group = parser.add_argument_group('Speech')
#     # Options most relevant to speech
#     parser.add_argument('--sample_rate', '-sample_rate', type=int, default=16000,
#               help="Sample rate.")
#     parser.add_argument('--window_size', '-window_size', type=float, default=.02,
#               help="Window size for spectrogram in seconds.")
#
#     # Option most relevant to image input
#     parser.add_argument('--image_channel_size', '-image_channel_size',
#               type=int, default=3, choices=[3, 1],
#               help="Using grayscale image can training "
#                    "model faster and smaller")
#
#
# def translate_opts(parser):
#     """ Translation / inference options """
#     group = parser.add_argument_group('Model')
#     parser.add_argument('--model', '-model', dest='models', metavar='MODEL',
#               nargs='+', type=str, default=[], required=True,
#               help="Path to model .pt file(s). "
#                    "Multiple models can be specified, "
#                    "for ensemble decoding.")
#     parser.add_argument('--fp32', '-fp32', action='store_true',
#               help="Force the model to be in FP32 "
#                    "because FP16 is very slow on GTX1080(ti).")
#     parser.add_argument('--avg_raw_probs', '-avg_raw_probs', action='store_true',
#               help="If this is set, during ensembling scores from "
#                    "different models will be combined by averaging their "
#                    "raw probabilities and then taking the log. Otherwise, "
#                    "the log probabilities will be averaged directly. "
#                    "Necessary for models whose output layers can assign "
#                    "zero probability.")
#
#     group = parser.add_argument_group('Data')
#     parser.add_argument('--data_type', '-data_type', default="text",
#               help="Type of the source input. Options: [text|img].")
#
#     parser.add_argument('--src', '-src', required=True,
#               help="Source sequence to decode (one line per "
#                    "sequence)")
#     parser.add_argument('--src_dir', '-src_dir', default="",
#               help='Source directory for image or audio files')
#     parser.add_argument('--tgt', '-tgt',
#               help='True target sequence (optional)')
#     parser.add_argument('--shard_size', '-shard_size', type=int, default=10000,
#               help="Divide src and tgt (if applicable) into "
#                    "smaller multiple src and tgt files, then "
#                    "build shards, each shard will have "
#                    "opt.shard_size samples except last shard. "
#                    "shard_size=0 means no segmentation "
#                    "shard_size>0 means segment dataset into multiple shards, "
#                    "each shard has shard_size samples")
#     parser.add_argument('--output', '-output', default='pred.txt',
#               help="Path to output the predictions (each line will "
#                    "be the decoded sequence")
#     parser.add_argument('--report_bleu', '-report_bleu', action='store_true',
#               help="Report bleu score after translation, "
#                    "call tools/multi-bleu.perl on command line")
#     parser.add_argument('--report_rouge', '-report_rouge', action='store_true',
#               help="Report rouge 1/2/3/L/SU4 score after translation "
#                    "call tools/test_rouge.py on command line")
#     parser.add_argument('--report_time', '-report_time', action='store_true',
#               help="Report some translation time metrics")
#
#     # Options most relevant to summarization.
#     parser.add_argument('--dynamic_dict', '-dynamic_dict', action='store_true',
#               help="Create dynamic dictionaries")
#     parser.add_argument('--share_vocab', '-share_vocab', action='store_true',
#               help="Share source and target vocabulary")
#
#     group = parser.add_argument_group('Random Sampling')
#     parser.add_argument('--random_sampling_topk', '-random_sampling_topk',
#               default=1, type=int,
#               help="Set this to -1 to do random sampling from full "
#                    "distribution. Set this to value k>1 to do random "
#                    "sampling restricted to the k most likely next tokens. "
#                    "Set this to 1 to use argmax or for doing beam "
#                    "search.")
#     parser.add_argument('--random_sampling_temp', '-random_sampling_temp',
#               default=1., type=float,
#               help="If doing random sampling, divide the logits by "
#                    "this before computing softmax during decoding.")
#     parser.add_argument('--seed', '-seed', type=int, default=829,
#               help="Random seed")
#
#     group = parser.add_argument_group('Beam')
#     parser.add_argument('--beam_size', '-beam_size', type=int, default=5,
#               help='Beam size')
#     parser.add_argument('--min_length', '-min_length', type=int, default=0,
#               help='Minimum prediction length')
#     parser.add_argument('--max_length', '-max_length', type=int, default=100,
#               help='Maximum prediction length.')
#     parser.add_argument('--max_sent_length', '-max_sent_length', action=DeprecateAction,
#               help="Deprecated, use `-max_length` instead")
#
#     # Alpha and Beta values for Google Length + Coverage penalty
#     # Described here: https://arxiv.org/pdf/1609.08144.pdf, Section 7
#     parser.add_argument('--stepwise_penalty', '-stepwise_penalty', action='store_true',
#               help="Apply penalty at every decoding step. "
#                    "Helpful for summary penalty.")
#     parser.add_argument('--length_penalty', '-length_penalty', default='none',
#               choices=['none', 'wu', 'avg'],
#               help="Length Penalty to use.")
#     parser.add_argument('--ratio', '-ratio', type=float, default=-0.,
#               help="Ratio based beam stop condition")
#     parser.add_argument('--coverage_penalty', '-coverage_penalty', default='none',
#               choices=['none', 'wu', 'summary'],
#               help="Coverage Penalty to use.")
#     parser.add_argument('--alpha', '-alpha', type=float, default=0.,
#               help="Google NMT length penalty parameter "
#                    "(higher = longer generation)")
#     parser.add_argument('--beta', '-beta', type=float, default=-0.,
#               help="Coverage penalty parameter")
#     parser.add_argument('--block_ngram_repeat', '-block_ngram_repeat',
#               type=int, default=0,
#               help='Block repetition of ngrams during decoding.')
#     parser.add_argument('--ignore_when_blocking', '-ignore_when_blocking',
#               nargs='+', type=str, default=[],
#               help="Ignore these strings when blocking repeats. "
#                    "You want to block sentence delimiters.")
#     parser.add_argument('--replace_unk', '-replace_unk', action="store_true",
#               help="Replace the generated UNK tokens with the "
#                    "source token that had highest attention weight. If "
#                    "phrase_table is provided, it will look up the "
#                    "identified source token and give the corresponding "
#                    "target token. If it is not provided (or the identified "
#                    "source token does not exist in the table), then it "
#                    "will copy the source token.")
#     parser.add_argument('--phrase_table', '-phrase_table', type=str, default="",
#               help="If phrase_table is provided (with replace_unk), it will "
#                    "look up the identified source token and give the "
#                    "corresponding target token. If it is not provided "
#                    "(or the identified source token does not exist in "
#                    "the table), then it will copy the source token.")
#     group = parser.add_argument_group('Logging')
#     parser.add_argument('--verbose', '-verbose', action="store_true",
#               help='Print scores and predictions for each sentence')
#     parser.add_argument('--log_file', '-log_file', type=str, default="",
#               help="Output logs to a file under this path.")
#     parser.add_argument('--log_file_level', '-log_file_level', type=str,
#               action=StoreLoggingLevelAction,
#               choices=StoreLoggingLevelAction.CHOICES,
#               default="0")
#     parser.add_argument('--attn_debug', '-attn_debug', action="store_true",
#               help='Print best attn for each word')
#     parser.add_argument('--dump_beam', '-dump_beam', type=str, default="",
#               help='File to dump beam information to.')
#     parser.add_argument('--n_best', '-n_best', type=int, default=1,
#               help="If verbose is set, will output the n_best "
#                    "decoded sentences")
#
#     group = parser.add_argument_group('Efficiency')
#     parser.add_argument('--batch_size', '-batch_size', type=int, default=30,
#               help='Batch size')
#     parser.add_argument('--gpu', '-gpu', type=int, default=-1,
#               help="Device to run on")
#
#     # Options most relevant to speech.
#     group = parser.add_argument_group('Speech')
#     parser.add_argument('--sample_rate', '-sample_rate', type=int, default=16000,
#               help="Sample rate.")
#     parser.add_argument('--window_size', '-window_size', type=float, default=.02,
#               help='Window size for spectrogram in seconds')
#     parser.add_argument('--window_stride', '-window_stride', type=float, default=.01,
#               help='Window stride for spectrogram in seconds')
#     parser.add_argument('--window', '-window', default='hamming',
#               help='Window type for spectrogram generation')
#
#     # Option most relevant to image input
#     parser.add_argument('--image_channel_size', '-image_channel_size',
#               type=int, default=3, choices=[3, 1],
#               help="Using grayscale image can training "
#                    "model faster and smaller")


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture('opennmt', 'opennmt')
def base_architecture(args):
    # Optimization Options
    args.dropout = getattr(args, 'dropout', 0.3)

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

    args.pre_word_vecs_enc = getattr(args, 'pre_word_vecs_enc', None)
    args.pre_word_vecs_dec = getattr(args, 'pre_word_vecs_dec', None)
    args.fix_word_vecs_enc = getattr(args, 'fix_word_vecs_enc', False)
    args.fix_word_vecs_dec = getattr(args, 'fix_word_vecs_dec', False)

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

    args.context_gate = getattr(args, 'context_gate', None)
    args.pre_word_vecs_dec = getattr(args, 'pre_word_vecs_dec', None)

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
                -batch_size 16 \
                -valid_batch_size 16 \
                -optim adagrad \
                -learning_rate 0.15 \
                -adagrad_accumulator_init 0.1 \
                -copy_loss_by_seqlength \
                -seed 777 \
                -world_size 2
    '''
    args.dropout = getattr(args, 'dropout', 0.0)
    args.word_vec_size = getattr(args, 'word_vec_size', 128)
    args.encoder_type = getattr(args, 'encoder_type', 'brnn')
    args.layers = getattr(args, 'layers', 1)
    args.rnn_size = getattr(args, 'rnn_size', 512)
    args.bridge = getattr(args, 'bridge', True)
    args.global_attention = getattr(args, 'global_attention', 'mlp')
    args.copy_attn = getattr(args, 'copy_attn', True)
    args.reuse_copy_attn = getattr(args, 'reuse_copy_attn', True)


@register_model_architecture('opennmt', 'opennmt_cnndm_transformer')
def opennmt_cnndm_transformer(args):
    '''
                   -max_grad_norm 0 \
                   -optim adam \
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
                   -world_size 2
    '''
    args.dropout = getattr(args, 'dropout', 0.2)
    args.word_vec_size = getattr(args, 'word_vec_size', 512)
    args.share_embeddings = getattr(args, 'share_embeddings', True)
    args.position_encoding = getattr(args, 'position_encoding', True)
    args.encoder_type = getattr(args, 'encoder_type', 'transformer')
    args.decoder_type = getattr(args, 'decoder_type', 'transformer')
    args.layers = getattr(args, 'layers', 4)
    args.rnn_size = getattr(args, 'rnn_size', 512)
    args.copy_attn = getattr(args, 'copy_attn', True)


@register_model_architecture('opennmt', 'opennmt_gigaword')
def opennmt_gigaword(args):
    '''
                -train_steps 200000
    '''
    args.copy_attn = getattr(args, 'copy_attn', True)
    args.reuse_copy_attn = getattr(args, 'reuse_copy_attn', True)
