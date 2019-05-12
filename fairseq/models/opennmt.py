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

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from fairseq import options, utils
from fairseq.modules import (
    AdaptiveInput, AdaptiveSoftmax, CharacterTokenEmbedder, LayerNorm,
    LearnedPositionalEmbedding, MultiheadAttention, SinusoidalPositionalEmbedding,
)

from fairseq.models.sru import CheckSRU

from . import (
    FairseqIncrementalDecoder, FairseqEncoder, FairseqLanguageModel,
    FairseqModel, register_model, register_model_architecture,
    SRU,
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
        # fmt: off
        # config_opts(parser)
        # model_opts(parser)
        # train_opts(parser)
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        # def validate_train_opts(args):
        #     if args.truncated_decoder > 0 and max(args.accum_count) > 1:
        #         raise AssertionError("BPTT is not compatible with -accum > 1")
        #
        # def update_model_opts(args):
        #     if args.word_vec_size > 0:
        #         args.src_word_vec_size = args.word_vec_size
        #         args.tgt_word_vec_size = args.word_vec_size
        #
        #     if args.layers > 0:
        #         args.enc_layers = args.layers
        #         args.dec_layers = args.layers
        #
        #     if args.rnn_size > 0:
        #         args.enc_rnn_size = args.rnn_size
        #         args.dec_rnn_size = args.rnn_size
        #
        #     if args.copy_attn_type is None:
        #         args.copy_attn_type = args.global_attention
        #
        # def validate_model_opts(args):
        #     assert args.model_type in ["text", "img", "audio"], \
        #         "Unsupported model type %s" % args.model_type
        #
        #     # this check is here because audio allows the encoder and decoder to
        #     # be different sizes, but other model types do not yet
        #     same_size = args.enc_rnn_size == args.dec_rnn_size
        #     assert args.model_type == 'audio' or same_size, \
        #         "The encoder and decoder rnns must be the same size for now"
        #
        #     assert args.rnn_type != "SRU" or args.device_id, \
        #         "Using SRU requires --device-id set."
        #     if args.share_embeddings:
        #         if args.model_type != "text":
        #             raise AssertionError(
        #                 "--share_embeddings requires --model_type text.")
        #
        # validate_train_opts(args)
        # update_model_opts(args)
        # validate_model_opts(args)

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

        def build_encoder(args, dictionary, embed_tokens):
            """
            Various encoder dispatcher function.
            Args:
                args: the option in current environment.
                embeddings (Embeddings): vocab embeddings for this encoder.
            """
            enc_type = args.encoder_type if args.model_type == "text" else args.model_type
            return str2enc[enc_type].from_opt(args, dictionary, embed_tokens)

        # encoder = TransformerEncoder(args, src_dict, encoder_embed_tokens)
        encoder = build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = TransformerDecoder(args, tgt_dict, decoder_embed_tokens)
        return TransformerModel(encoder, decoder)


class RNNEncoder(FairseqEncoder):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, dictionary=None, embed_tokens=None,
                 use_bridge=False):
        super().__init__(dictionary)
        assert embed_tokens is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embed_tokens

        def rnn_factory(rnn_type, **kwargs):
            """ rnn factory, Use pytorch version when available. """
            no_pack_padded_seq = False
            if rnn_type == "SRU":
                # SRU doesn't support PackedSequence.
                no_pack_padded_seq = True
                rnn = SRU(**kwargs)
            else:
                rnn = getattr(nn, rnn_type)(**kwargs)
            return rnn, no_pack_padded_seq

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embed_tokens.embedding_dim,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, dictionary, embed_tokens):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout,
            dictionary,
            embed_tokens,
            opt.bridge)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs


class CNNEncoder(FairseqEncoder):
    """Encoder based on "Convolutional Sequence to Sequence Learning"
    :cite:`DBLP:journals/corr/GehringAGYD17`.
    """

    def __init__(self, num_layers, hidden_size,
                 cnn_kernel_width, dropout, embeddings):
        super(CNNEncoder, self).__init__()

        self.embeddings = embeddings
        input_size = embeddings.embedding_size
        self.linear = nn.Linear(input_size, hidden_size)
        self.cnn = StackedCNN(num_layers, hidden_size,
                              cnn_kernel_width, dropout)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.cnn_kernel_width,
            opt.dropout,
            embeddings)

    def forward(self, input, lengths=None, hidden=None):
        """See :class:`onmt.modules.EncoderBase.forward()`"""
        self._check_args(input, lengths, hidden)

        emb = self.embeddings(input)
        # s_len, batch, emb_dim = emb.size()

        emb = emb.transpose(0, 1).contiguous()
        emb_reshape = emb.view(emb.size(0) * emb.size(1), -1)
        emb_remap = self.linear(emb_reshape)
        emb_remap = emb_remap.view(emb.size(0), emb.size(1), -1)
        emb_remap = shape_transform(emb_remap)
        out = self.cnn(emb_remap)

        return emb_remap.squeeze(3).transpose(0, 1).contiguous(), \
            out.squeeze(3).transpose(0, 1).contiguous(), lengths


class TransformerEncoder(FairseqEncoder):
    """The Transformer encoder from "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    .. mermaid::

       graph BT
          A[input]
          B[multi-head self-attn]
          C[feed forward]
          O[output]
          A --> B
          B --> C
          C --> O

    Args:
        num_layers (int): number of encoder layers
        d_model (int): size of the model
        heads (int): number of heads
        d_ff (int): size of the inner FF layer
        dropout (float): dropout parameters
        embeddings (onmt.modules.Embeddings):
          embeddings to use, should have positional encodings

    Returns:
        (torch.FloatTensor, torch.FloatTensor):

        * embeddings ``(src_len, batch_size, model_dim)``
        * memory_bank ``(src_len, batch_size, model_dim)``
    """

    def __init__(self, num_layers, d_model, heads, d_ff, dropout, embeddings,
                 max_relative_positions):
        super(TransformerEncoder, self).__init__()

        self.embeddings = embeddings
        self.transformer = nn.ModuleList(
            [TransformerEncoderLayer(
                d_model, heads, d_ff, dropout,
                max_relative_positions=max_relative_positions)
             for i in range(num_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.heads,
            opt.transformer_ff,
            opt.dropout,
            embeddings,
            opt.max_relative_positions)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)

        out = emb.transpose(0, 1).contiguous()
        words = src[:, :, 0].transpose(0, 1)
        w_batch, w_len = words.size()
        padding_idx = self.embeddings.word_padding_idx
        mask = words.data.eq(padding_idx).unsqueeze(1)  # [B, 1, T]
        # Run the forward pass of every layer of the tranformer.
        for layer in self.transformer:
            out = layer(out, mask)
        out = self.layer_norm(out)

        return emb, out.transpose(0, 1).contiguous(), lengths


class ImageEncoder(FairseqEncoder):
    """A simple encoder CNN -> RNN for image src.

    Args:
        num_layers (int): number of encoder layers.
        bidirectional (bool): bidirectional encoder.
        rnn_size (int): size of hidden states of the rnn.
        dropout (float): dropout probablity.
    """

    def __init__(self, num_layers, bidirectional, rnn_size, dropout,
                 image_chanel_size=3):
        super(ImageEncoder, self).__init__()
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = rnn_size

        self.layer1 = nn.Conv2d(image_chanel_size, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)

        src_size = 512
        self.rnn = nn.LSTM(src_size, int(rnn_size / self.num_directions),
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=bidirectional)
        self.pos_lut = nn.Embedding(1000, src_size)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        if embeddings is not None:
            raise ValueError("Cannot use embeddings with ImageEncoder.")
        # why is the model_opt.__dict__ check necessary?
        if "image_channel_size" not in opt.__dict__:
            image_channel_size = 3
        else:
            image_channel_size = opt.image_channel_size
        return cls(
            opt.enc_layers,
            opt.brnn,
            opt.enc_rnn_size,
            opt.dropout,
            image_channel_size
        )

    def load_pretrained_vectors(self, opt):
        """Pass in needed options only when modify function definition."""
        pass

    def forward(self, src, lengths=None):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""

        batch_size = src.size(0)
        # (batch_size, 64, imgH, imgW)
        # layer 1
        src = F.relu(self.layer1(src[:, :, :, :] - 0.5), True)

        # (batch_size, 64, imgH/2, imgW/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        # (batch_size, 128, imgH/2, imgW/2)
        # layer 2
        src = F.relu(self.layer2(src), True)

        # (batch_size, 128, imgH/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))

        #  (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer 3
        # batch norm 1
        src = F.relu(self.batch_norm1(self.layer3(src)), True)

        # (batch_size, 256, imgH/2/2, imgW/2/2)
        # layer4
        src = F.relu(self.layer4(src), True)

        # (batch_size, 256, imgH/2/2/2, imgW/2/2)
        src = F.max_pool2d(src, kernel_size=(1, 2), stride=(1, 2))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2)
        # layer 5
        # batch norm 2
        src = F.relu(self.batch_norm2(self.layer5(src)), True)

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))

        # (batch_size, 512, imgH/2/2/2, imgW/2/2/2)
        src = F.relu(self.batch_norm3(self.layer6(src)), True)

        # # (batch_size, 512, H, W)
        all_outputs = []
        for row in range(src.size(2)):
            inp = src[:, :, row, :].transpose(0, 2) \
                .transpose(1, 2)
            row_vec = torch.Tensor(batch_size).type_as(inp.data) \
                .long().fill_(row)
            pos_emb = self.pos_lut(row_vec)
            with_pos = torch.cat(
                (pos_emb.view(1, pos_emb.size(0), pos_emb.size(1)), inp), 0)
            outputs, hidden_t = self.rnn(with_pos)
            all_outputs.append(outputs)
        out = torch.cat(all_outputs, 0)

        return hidden_t, out, lengths


class AudioEncoder(FairseqEncoder):
    """A simple encoder CNN -> RNN for audio input.

    Args:
        rnn_type (str): Type of RNN (e.g. GRU, LSTM, etc).
        enc_layers (int): Number of encoder layers.
        dec_layers (int): Number of decoder layers.
        brnn (bool): Bidirectional encoder.
        enc_rnn_size (int): Size of hidden states of the rnn.
        dec_rnn_size (int): Size of the decoder hidden states.
        enc_pooling (str): A comma separated list either of length 1
            or of length ``enc_layers`` specifying the pooling amount.
        dropout (float): dropout probablity.
        sample_rate (float): input spec
        window_size (int): input spec
    """

    def __init__(self, rnn_type, enc_layers, dec_layers, brnn,
                 enc_rnn_size, dec_rnn_size, enc_pooling, dropout,
                 sample_rate, window_size):
        super(AudioEncoder, self).__init__()
        self.enc_layers = enc_layers
        self.rnn_type = rnn_type
        self.dec_layers = dec_layers
        num_directions = 2 if brnn else 1
        self.num_directions = num_directions
        assert enc_rnn_size % num_directions == 0
        enc_rnn_size_real = enc_rnn_size // num_directions
        assert dec_rnn_size % num_directions == 0
        self.dec_rnn_size = dec_rnn_size
        dec_rnn_size_real = dec_rnn_size // num_directions
        self.dec_rnn_size_real = dec_rnn_size_real
        self.dec_rnn_size = dec_rnn_size
        input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        enc_pooling = enc_pooling.split(',')
        assert len(enc_pooling) == enc_layers or len(enc_pooling) == 1
        if len(enc_pooling) == 1:
            enc_pooling = enc_pooling * enc_layers
        enc_pooling = [int(p) for p in enc_pooling]
        self.enc_pooling = enc_pooling

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.W = nn.Linear(enc_rnn_size, dec_rnn_size, bias=False)
        self.batchnorm_0 = nn.BatchNorm1d(enc_rnn_size, affine=True)
        self.rnn_0, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=input_size,
                        hidden_size=enc_rnn_size_real,
                        num_layers=1,
                        dropout=dropout,
                        bidirectional=brnn)
        self.pool_0 = nn.MaxPool1d(enc_pooling[0])
        for l in range(enc_layers - 1):
            batchnorm = nn.BatchNorm1d(enc_rnn_size, affine=True)
            rnn, _ = \
                rnn_factory(rnn_type,
                            input_size=enc_rnn_size,
                            hidden_size=enc_rnn_size_real,
                            num_layers=1,
                            dropout=dropout,
                            bidirectional=brnn)
            setattr(self, 'rnn_%d' % (l + 1), rnn)
            setattr(self, 'pool_%d' % (l + 1),
                    nn.MaxPool1d(enc_pooling[l + 1]))
            setattr(self, 'batchnorm_%d' % (l + 1), batchnorm)

    @classmethod
    def from_opt(cls, opt, embeddings=None):
        """Alternate constructor."""
        if embeddings is not None:
            raise ValueError("Cannot use embeddings with AudioEncoder.")
        return cls(
            opt.rnn_type,
            opt.enc_layers,
            opt.dec_layers,
            opt.brnn,
            opt.enc_rnn_size,
            opt.dec_rnn_size,
            opt.audio_enc_pooling,
            opt.dropout,
            opt.sample_rate,
            opt.window_size)

    def forward(self, src, lengths=None):
        """See :func:`onmt.encoders.encoder.EncoderBase.forward()`"""
        batch_size, _, nfft, t = src.size()
        src = src.transpose(0, 1).transpose(0, 3).contiguous() \
                 .view(t, batch_size, nfft)
        orig_lengths = lengths
        lengths = lengths.view(-1).tolist()

        for l in range(self.enc_layers):
            rnn = getattr(self, 'rnn_%d' % l)
            pool = getattr(self, 'pool_%d' % l)
            batchnorm = getattr(self, 'batchnorm_%d' % l)
            stride = self.enc_pooling[l]
            packed_emb = pack(src, lengths)
            memory_bank, tmp = rnn(packed_emb)
            memory_bank = unpack(memory_bank)[0]
            t, _, _ = memory_bank.size()
            memory_bank = memory_bank.transpose(0, 2)
            memory_bank = pool(memory_bank)
            lengths = [int(math.floor((length - stride) / stride + 1))
                       for length in lengths]
            memory_bank = memory_bank.transpose(0, 2)
            src = memory_bank
            t, _, num_feat = src.size()
            src = batchnorm(src.contiguous().view(-1, num_feat))
            src = src.view(t, -1, num_feat)
            if self.dropout and l + 1 != self.enc_layers:
                src = self.dropout(src)

        memory_bank = memory_bank.contiguous().view(-1, memory_bank.size(2))
        memory_bank = self.W(memory_bank).view(-1, batch_size,
                                               self.dec_rnn_size)

        state = memory_bank.new_full((self.dec_layers * self.num_directions,
                                      batch_size, self.dec_rnn_size_real), 0)
        if self.rnn_type == 'LSTM':
            # The encoder hidden is  (layers*directions) x batch x dim.
            encoder_final = (state, state)
        else:
            encoder_final = state
        return encoder_final, memory_bank, orig_lengths.new_tensor(lengths)


class MeanEncoder(FairseqEncoder):
    """A trivial non-recurrent encoder. Simply applies mean pooling.

    Args:
       num_layers (int): number of replicated layers
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, num_layers, embeddings):
        super(MeanEncoder, self).__init__()
        self.num_layers = num_layers
        self.embeddings = embeddings

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.enc_layers,
            embeddings)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        _, batch, emb_dim = emb.size()

        if lengths is not None:
            # we avoid padding while mean pooling
            mask = sequence_mask(lengths).float()
            mask = mask / lengths.unsqueeze(1).float()
            mean = torch.bmm(mask.unsqueeze(1), emb.transpose(0, 1)).squeeze(1)
        else:
            mean = emb.mean(0)

        mean = mean.expand(self.num_layers, batch, emb_dim)
        memory_bank = emb
        encoder_final = (mean, mean)
        return encoder_final, memory_bank, lengths


class TransformerEncoderLayer(nn.Module):
    """
    A single layer of the transformer encoder.

    Args:
        d_model (int): the dimension of keys/values/queries in
                   MultiHeadedAttention, also the input size of
                   the first-layer of the PositionwiseFeedForward.
        heads (int): the number of head for MultiHeadedAttention.
        d_ff (int): the second-layer of the PositionwiseFeedForward.
        dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, heads, d_ff, dropout,
                 max_relative_positions=0):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout,
            max_relative_positions=max_relative_positions)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, mask):
        """
        Args:
            inputs (FloatTensor): ``(batch_size, src_len, model_dim)``
            mask (LongTensor): ``(batch_size, src_len, src_len)``

        Returns:
            (FloatTensor):

            * outputs ``(batch_size, src_len, model_dim)``
        """
        input_norm = self.layer_norm(inputs)
        context, _ = self.self_attn(input_norm, input_norm, input_norm,
                                    mask=mask, type="self")
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


# def config_opts(parser):
#     parser.add_argument('--config', '-config', required=False,
#                         is_config_file_arg=True, help='config file path')
#     parser.add_argument('--save_config', '-save_config', required=False,
#                         is_write_out_config_file_arg=True,
#                         help='config file save path')
#
#
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
#
#
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
#
#
# def build_embeddings(args, dictionary, for_encoder=True):
#     emb_dim = args.src_word_vec_size if for_encoder else args.tgt_word_vec_size
#
#     word_padding_idx = dictionary.pad()
#
#     num_word_embeddings = len(dictionary)
#
#     fix_word_vecs = args.fix_word_vecs_enc if for_encoder \
#         else args.fix_word_vecs_dec
#
#     emb = Embeddings(
#         word_vec_size=emb_dim,
#         position_encoding=args.position_encoding,
#         feat_merge=args.feat_merge,
#         feat_vec_exponent=args.feat_vec_exponent,
#         feat_vec_size=args.feat_vec_size,
#         dropout=args.dropout,
#         word_padding_idx=word_padding_idx,
#         feat_padding_idx=[],
#         word_vocab_size=num_word_embeddings,
#         feat_vocab_sizes=[],
#         sparse=args.optim == "sparseadam",
#         fix_word_vecs=fix_word_vecs
#     )
#     return emb
#
#
# def build_encoder(opt, embeddings):
#     """
#     Various encoder dispatcher function.
#     Args:
#         opt: the option in current environment.
#         embeddings (Embeddings): vocab embeddings for this encoder.
#     """
#     enc_type = opt.encoder_type if opt.model_type == "text" else opt.model_type
#     return str2enc[enc_type].from_opt(opt, embeddings)
#
#
# def build_decoder(opt, embeddings):
#     """
#     Various decoder dispatcher function.
#     Args:
#         opt: the option in current environment.
#         embeddings (Embeddings): vocab embeddings for this decoder.
#     """
#     dec_type = "ifrnn" if opt.decoder_type == "rnn" and opt.input_feed \
#                else opt.decoder_type
#     return str2dec[dec_type].from_opt(opt, embeddings)
#
#
# def build_base_model(args, task):
#     # Build embeddings.
#     src_emb = build_embeddings(args, task.source_dictionary)
#
#     # Build encoder.
#     encoder = build_encoder(args, src_emb)
#
#     # Build decoder.
#     tgt_field = task["tgt"]
#     tgt_emb = build_embeddings(args, tgt_field, for_encoder=False)
#
#     # Share the embedding matrix - preprocess with share_vocab required.
#     if args.share_embeddings:
#         # src/tgt vocab should be the same if `-share_vocab` is specified.
#         assert src_field.base_field.vocab == tgt_field.base_field.vocab, \
#             "preprocess with -share_vocab if you use share_embeddings"
#
#         tgt_emb.word_lut.weight = src_emb.word_lut.weight
#
#     decoder = build_decoder(args, tgt_emb)
#
#     # Build NMTModel(= encoder + decoder).
#     model = onmt.models.NMTModel(encoder, decoder)
#
#     # Build Generator.
#     if not args.copy_attn:
#         if args.generator_function == "sparsemax":
#             gen_func = onmt.modules.sparse_activations.LogSparsemax(dim=-1)
#         else:
#             gen_func = nn.LogSoftmax(dim=-1)
#         generator = nn.Sequential(
#             nn.Linear(args.dec_rnn_size,
#                       len(task["tgt"].base_field.vocab)),
#             Cast(torch.float32),
#             gen_func
#         )
#         if args.share_decoder_embeddings:
#             generator[0].weight = decoder.embeddings.word_lut.weight
#     else:
#         tgt_base_field = task["tgt"].base_field
#         vocab_size = len(tgt_base_field.vocab)
#         pad_idx = tgt_base_field.vocab.stoi[tgt_base_field.pad_token]
#         generator = CopyGenerator(args.dec_rnn_size, vocab_size, pad_idx)
#
#     if args.param_init != 0.0:
#         for p in model.parameters():
#             p.data.uniform_(-args.param_init, args.param_init)
#         for p in generator.parameters():
#             p.data.uniform_(-args.param_init, args.param_init)
#     if args.param_init_glorot:
#         for p in model.parameters():
#             if p.dim() > 1:
#                 xavier_uniform_(p)
#         for p in generator.parameters():
#             if p.dim() > 1:
#                 xavier_uniform_(p)
#
#     if hasattr(model.encoder, 'embeddings'):
#         model.encoder.embeddings.load_pretrained_vectors(
#             args.pre_word_vecs_enc)
#     if hasattr(model.decoder, 'embeddings'):
#         model.decoder.embeddings.load_pretrained_vectors(
#             args.pre_word_vecs_dec)
#
#     model.generator = generator
#     if args.model_dtype == 'fp16':
#         model.half()
#
#     return model


# class PositionalEncoding(nn.Module):
#     """Sinusoidal positional encoding for non-recurrent neural networks.
#
#     Implementation based on "Attention Is All You Need"
#     :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`
#
#     Args:
#        dropout (float): dropout parameter
#        dim (int): embedding size
#     """
#
#     def __init__(self, dropout, dim, max_len=5000):
#         if dim % 2 != 0:
#             raise ValueError("Cannot use sin/cos positional encoding with "
#                              "odd dim (got dim={:d})".format(dim))
#         pe = torch.zeros(max_len, dim)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
#                              -(math.log(10000.0) / dim)))
#         pe[:, 0::2] = torch.sin(position.float() * div_term)
#         pe[:, 1::2] = torch.cos(position.float() * div_term)
#         pe = pe.unsqueeze(1)
#         super(PositionalEncoding, self).__init__()
#         self.register_buffer('pe', pe)
#         self.dropout = nn.Dropout(p=dropout)
#         self.dim = dim
#
#     def forward(self, emb, step=None):
#         """Embed inputs.
#
#         Args:
#             emb (FloatTensor): Sequence of word vectors
#                 ``(seq_len, batch_size, self.dim)``
#             step (int or NoneType): If stepwise (``seq_len = 1``), use
#                 the encoding for this position.
#         """
#
#         emb = emb * math.sqrt(self.dim)
#         if step is None:
#             emb = emb + self.pe[:emb.size(0)]
#         else:
#             emb = emb + self.pe[step]
#         emb = self.dropout(emb)
#         return emb


# class Embeddings(nn.Module):
#     """Words embeddings for encoder/decoder.
#
#     Additionally includes ability to add sparse input features
#     based on "Linguistic Input Features Improve Neural Machine Translation"
#     :cite:`sennrich2016linguistic`.
#
#
#     .. mermaid::
#
#        graph LR
#           A[Input]
#           C[Feature 1 Lookup]
#           A-->B[Word Lookup]
#           A-->C
#           A-->D[Feature N Lookup]
#           B-->E[MLP/Concat]
#           C-->E
#           D-->E
#           E-->F[Output]
#
#     Args:
#         word_vec_size (int): size of the dictionary of embeddings.
#         word_padding_idx (int): padding index for words in the embeddings.
#         feat_padding_idx (List[int]): padding index for a list of features
#                                    in the embeddings.
#         word_vocab_size (int): size of dictionary of embeddings for words.
#         feat_vocab_sizes (List[int], optional): list of size of dictionary
#             of embeddings for each feature.
#         position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
#         feat_merge (string): merge action for the features embeddings:
#             concat, sum or mlp.
#         feat_vec_exponent (float): when using `-feat_merge concat`, feature
#             embedding size is N^feat_dim_exponent, where N is the
#             number of values the feature takes.
#         feat_vec_size (int): embedding dimension for features when using
#             `-feat_merge mlp`
#         dropout (float): dropout probability.
#     """
#
#     def __init__(self, word_vec_size,
#                  word_vocab_size,
#                  word_padding_idx,
#                  position_encoding=False,
#                  feat_merge="concat",
#                  feat_vec_exponent=0.7,
#                  feat_vec_size=-1,
#                  feat_padding_idx=[],
#                  feat_vocab_sizes=[],
#                  dropout=0,
#                  sparse=False,
#                  fix_word_vecs=False):
#         self._validate_args(feat_merge, feat_vocab_sizes, feat_vec_exponent,
#                             feat_vec_size, feat_padding_idx)
#
#         if feat_padding_idx is None:
#             feat_padding_idx = []
#         self.word_padding_idx = word_padding_idx
#
#         self.word_vec_size = word_vec_size
#
#         # Dimensions and padding for constructing the word embedding matrix
#         vocab_sizes = [word_vocab_size]
#         emb_dims = [word_vec_size]
#         pad_indices = [word_padding_idx]
#
#         # Dimensions and padding for feature embedding matrices
#         # (these have no effect if feat_vocab_sizes is empty)
#         if feat_merge == 'sum':
#             feat_dims = [word_vec_size] * len(feat_vocab_sizes)
#         elif feat_vec_size > 0:
#             feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
#         else:
#             feat_dims = [int(vocab ** feat_vec_exponent)
#                          for vocab in feat_vocab_sizes]
#         vocab_sizes.extend(feat_vocab_sizes)
#         emb_dims.extend(feat_dims)
#         pad_indices.extend(feat_padding_idx)
#
#         # The embedding matrix look-up tables. The first look-up table
#         # is for words. Subsequent ones are for features, if any exist.
#         emb_params = zip(vocab_sizes, emb_dims, pad_indices)
#         embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
#                       for vocab, dim, pad in emb_params]
#         emb_luts = Elementwise(feat_merge, embeddings)
#
#         # The final output size of word + feature vectors. This can vary
#         # from the word vector size if and only if features are defined.
#         # This is the attribute you should access if you need to know
#         # how big your embeddings are going to be.
#         self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
#                                else word_vec_size)
#
#         # The sequence of operations that converts the input sequence
#         # into a sequence of embeddings. At minimum this consists of
#         # looking up the embeddings for each word and feature in the
#         # input. Model parameters may require the sequence to contain
#         # additional operations as well.
#         super(Embeddings, self).__init__()
#         self.make_embedding = nn.Sequential()
#         self.make_embedding.add_module('emb_luts', emb_luts)
#
#         if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
#             in_dim = sum(emb_dims)
#             mlp = nn.Sequential(nn.Linear(in_dim, word_vec_size), nn.ReLU())
#             self.make_embedding.add_module('mlp', mlp)
#
#         self.position_encoding = position_encoding
#
#         if self.position_encoding:
#             pe = PositionalEncoding(dropout, self.embedding_size)
#             self.make_embedding.add_module('pe', pe)
#
#         if fix_word_vecs:
#             self.word_lut.weight.requires_grad = False
#
#     def _validate_args(self, feat_merge, feat_vocab_sizes, feat_vec_exponent,
#                        feat_vec_size, feat_padding_idx):
#         if feat_merge == "sum":
#             # features must use word_vec_size
#             if feat_vec_exponent != 0.7:
#                 warnings.warn("Merging with sum, but got non-default "
#                               "feat_vec_exponent. It will be unused.")
#             if feat_vec_size != -1:
#                 warnings.warn("Merging with sum, but got non-default "
#                               "feat_vec_size. It will be unused.")
#         elif feat_vec_size > 0:
#             # features will use feat_vec_size
#             if feat_vec_exponent != -1:
#                 warnings.warn("Not merging with sum and positive "
#                               "feat_vec_size, but got non-default "
#                               "feat_vec_exponent. It will be unused.")
#         else:
#             if feat_vec_exponent <= 0:
#                 raise ValueError("Using feat_vec_exponent to determine "
#                                  "feature vec size, but got feat_vec_exponent "
#                                  "less than or equal to 0.")
#         n_feats = len(feat_vocab_sizes)
#         if n_feats != len(feat_padding_idx):
#             raise ValueError("Got unequal number of feat_vocab_sizes and "
#                              "feat_padding_idx ({:d} != {:d})".format(
#                                 n_feats, len(feat_padding_idx)))
#
#     @property
#     def word_lut(self):
#         """Word look-up table."""
#         return self.make_embedding[0][0]
#
#     @property
#     def emb_luts(self):
#         """Embedding look-up table."""
#         return self.make_embedding[0]
#
#     def load_pretrained_vectors(self, emb_file):
#         """Load in pretrained embeddings.
#
#         Args:
#           emb_file (str) : path to torch serialized embeddings
#         """
#
#         if emb_file:
#             pretrained = torch.load(emb_file)
#             pretrained_vec_size = pretrained.size(1)
#             if self.word_vec_size > pretrained_vec_size:
#                 self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
#             elif self.word_vec_size < pretrained_vec_size:
#                 self.word_lut.weight.data \
#                     .copy_(pretrained[:, :self.word_vec_size])
#             else:
#                 self.word_lut.weight.data.copy_(pretrained)
#
#     def forward(self, source, step=None):
#         """Computes the embeddings for words and features.
#
#         Args:
#             source (LongTensor): index tensor ``(len, batch, nfeat)``
#
#         Returns:
#             FloatTensor: Word embeddings ``(len, batch, embedding_size)``
#         """
#
#         if self.position_encoding:
#             for i, module in enumerate(self.make_embedding._modules.values()):
#                 if i == len(self.make_embedding._modules.values()) - 1:
#                     source = module(source, step=step)
#                 else:
#                     source = module(source)
#         else:
#             source = self.make_embedding(source)
#
#         return source

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


@register_model_architecture('opennmt', 'opennmt')
def base_architecture(args):
    # Training Options
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

    args.param_init = getattr(args, 'param_init', type=float, default=0.1,
              help="Parameters are initialized over uniform distribution "
                   "with support (-param_init, param_init). "
                   "Use 0 to not use initialization")
    args.param_init_glorot = getattr(args, 'param_init_glorot', action='store_true',
              help="Init parameters with xavier_uniform. "
                   "Required for transformer.")

    args.train_from = getattr(args, 'train_from', default='', type=str,
              help="If training from a checkpoint then this is the "
                   "path to the pretrained model's state_dict.")
    args.reset_optim = getattr(args, 'reset_optim', default='none',
              choices=['none', 'all', 'states', 'keep_states'],
              help="Optimization resetter when train_from.")

    # Pretrained word vectors
    args.pre_word_vecs_enc = getattr(args, 'pre_word_vecs_enc',
              help="If a valid path is specified, then this will load "
                   "pretrained word embeddings on the encoder side. "
                   "See README for specific formatting instructions.")
    args.pre_word_vecs_dec = getattr(args, 'pre_word_vecs_dec',
              help="If a valid path is specified, then this will load "
                   "pretrained word embeddings on the decoder side. "
                   "See README for specific formatting instructions.")
    # Fixed word vectors
    args.fix_word_vecs_enc = getattr(args, 'fix_word_vecs_enc',
              action='store_true',
              help="Fix word embeddings on the encoder side.")
    args.fix_word_vecs_dec = getattr(args, 'fix_word_vecs_dec',
              action='store_true',
              help="Fix word embeddings on the decoder side.")

    args.dropout = getattr(args, 'dropout', type=float, default=0.3,
                        help="Dropout probability; applied in LSTM stacks.")
    args.truncated_decoder = getattr(args, 'truncated_decoder', type=int, default=0,
              help="""Truncated bptt.""")


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
    args.dropout = getattr(args, 'dropout', 0.)
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
