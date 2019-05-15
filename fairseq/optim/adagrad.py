# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.optim

from . import FairseqOptimizer, register_optimizer


@register_optimizer('adagrad')
class Adagrad(FairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args, params)
        self._optimizer = torch.optim.Adagrad(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        parser.add_argument('--adagrad_accumulator_init', type=float, default=0,
                            help="Initializes the accumulator values in adagrad. "
                                 "Mirrors the initial_accumulator_value option "
                                 "in the tensorflow adagrad (use 0.1 for their default).")

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        return {
            'lr': self.args.lr[0],
            'weight_decay': self.args.weight_decay,
            'initial_accumulator_value': self.args.adagrad_accumulator_init,
        }
