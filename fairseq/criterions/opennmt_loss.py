# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch.nn.functional as F
import torch.nn as nn

from fairseq import utils

from . import FairseqCriterion, register_criterion

from onmt.modules import CopyGeneratorLoss, CopyGeneratorLossCompute
from onmt.utils.loss import LabelSmoothingLoss, NMTLossCompute
from onmt.modules.sparse_losses import SparsemaxLoss
from onmt.modules.sparse_activations import LogSparsemax


@register_criterion('opennmt_loss')
class OpenNMTLoss(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

        vocab = task.target_field.vocab
        unk_idx = task.target_dictionary.unk()
        model = task.model.opennmt_model

        # from onmt.utils.loss.py, build_loss_compute()
        if args.copy_attn:
            criterion = CopyGeneratorLoss(
                len(vocab),
                args.copy_attn_force,
                unk_index=unk_idx, ignore_index=self.padding_idx)
        elif args.label_smoothing > 0:  # TODO: add "and train" back
            criterion = LabelSmoothingLoss(
                args.label_smoothing, len(vocab), ignore_index=self.padding_idx
            )
        elif isinstance(model.generator[-1], LogSparsemax):
            criterion = SparsemaxLoss(ignore_index=self.padding_idx, reduction='sum')
        else:
            criterion = nn.NLLLoss(ignore_index=self.padding_idx, reduction='sum')

        # if the loss function operates on vectors of raw logits instead of
        # probabilities, only the first part of the generator needs to be
        # passed to the NMTLossCompute. At the moment, the only supported
        # loss function of this kind is the sparsemax loss.
        use_raw_logits = isinstance(criterion, SparsemaxLoss)
        loss_gen = model.generator[0] if use_raw_logits else model.generator
        if args.copy_attn:
            self.compute = CopyGeneratorLossCompute(
                criterion, loss_gen, vocab, args.copy_loss_by_seqlength
            )
        else:
            self.compute = NMTLossCompute(criterion, loss_gen)

        self.bptt = False

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label_smoothing', '-label_smoothing', type=float, default=0.0,
                  help="Label smoothing value epsilon. "
                       "Probabilities of all non-true labels "
                       "will be smoothed by epsilon / (vocab_size - 1). "
                       "Set to zero to turn off label smoothing. "
                       "For more detailed information, see: "
                       "https://arxiv.org/abs/1512.00567")

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        num_tokens = sample.tgt[1:, :, 0].ne(
            self.padding_idx).sum().item()
        if self.args.sentence_avg:
            sample_size = sample.tgt.size(1)
        else:
            sample_size = num_tokens

        outputs, attns = model(sample.src[0], sample.src[1], sample.tgt, bptt=self.bptt)
        self.bptt = True

        loss, batch_stats = self.compute(
            sample,
            outputs,
            attns,
            normalization=sample_size,
            shard_size=0,
            trunc_start=0,
            trunc_size=None)

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': num_tokens,
            'nsentences': sample.tgt.size(1),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
