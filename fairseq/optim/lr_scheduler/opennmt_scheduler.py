# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

from . import FairseqLRScheduler, register_lr_scheduler

from onmt.utils import Optimizer
from onmt.utils.optimizers import make_learning_rate_decay_fn


@register_lr_scheduler('opennmt_scheduler')
class OpenNMTSchedule(FairseqLRScheduler):
    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with OpenNMT.'
                ' Consider --lr-scheduler=fixed instead.'
            )

        self.optim = Optimizer(optimizer, args.lr,
                               learning_rate_decay_fn=make_learning_rate_decay_fn(args),
                               max_grad_norm=args.max_grad_norm)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        # = '--lr'
        # parser.add_argument('--learning_rate', '-learning_rate', type=float, default=1.0,
        #           help="Starting learning rate. "
        #                "Recommended settings: sgd = 1, adagrad = 0.1, "
        #                "adadelta = 1, adam = 0.001")
        parser.add_argument('--learning_rate_decay', '-learning_rate_decay',
                  type=float, default=0.5,
                  help="If update_learning_rate, decay learning rate by "
                       "this much if steps have gone past "
                       "start_decay_steps")
        parser.add_argument('--start_decay_steps', '-start_decay_steps',
                  type=int, default=50000,
                  help="Start decaying every decay_steps after "
                       "start_decay_steps")
        parser.add_argument('--decay_steps', '-decay_steps', type=int, default=10000,
                  help="Decay every decay_steps")

        parser.add_argument('--decay_method', '-decay_method', type=str, default="none",
                  choices=['noam', 'noamwd', 'rsqrt', 'none'],
                  help="Use a custom decay rate.")
        parser.add_argument('--warmup_steps', '-warmup_steps', type=int, default=4000,
                  help="Number of warmup steps for custom decay.")
        # fmt: on
