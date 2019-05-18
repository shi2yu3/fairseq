# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import os
import struct

import numpy as np
import torch
from torchtext.data.batch import Batch

from . import data_utils, FairseqDataset

from fairseq import utils


def collate(
    samples, pad_idx, eos_idx, input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, False, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source')
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target')
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens
    return batch


def generate_dummy_batch(num_tokens, collate_fn, dict, src_len=128, tgt_len=128):
    """Return a dummy batch with a given number of tokens."""
    bsz = num_tokens // max(src_len, tgt_len)
    return collate_fn([
        {
            'id': i,
            'source': dict.dummy_sentence(src_len),
            'target': dict.dummy_sentence(tgt_len),
        }
        for i in range(bsz)
    ])


class OpenNMTDataset(FairseqDataset):
    def __init__(self, iter, args, dict):
        super().__init__()
        self.iter = iter
        self.args = args
        self.dict = dict
        self.dataset = None

    # def read_data(self, path):
    #     self.dataset = torch.load(path)

    # def check_index(self, i):
    #     if i < 0 or i >= self.size:
    #         raise IndexError('index out of range')

    def __del__(self):
        if self.dataset:
            self.dataset.close()

    def __getitem__(self, i):
        if not self.dataset:
            self.read_data(self.path)
        self.check_index(i)
        tensor_size = int(self.sizes[self.dim_offsets[i]:self.dim_offsets[i + 1]])
        a = np.empty(tensor_size, dtype=self.dtype)
        self.dataset.seek(self.data_offsets[i] * self.element_size)
        self.dataset.readinto(a)
        item = torch.from_numpy(a).long()
        if self.fix_lua_indexing:
            item -= 1  # subtract 1 for 0-based indexing
        return item

    def __len__(self):
        return self.size

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one position
                    for input feeding/teacher forcing, of shape `(bsz,
                    tgt_len)`. This key will not be present if *input_feeding*
                    is ``False``. Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.dict.pad(), eos_idx=self.dict.eos(),
            input_feeding=self.args.input_feed,
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        """Return a dummy batch with a given number of tokens."""
        src_len, tgt_len = utils.resolve_max_positions(
            (src_len, tgt_len),
            max_positions,
            (self.args.src_seq_length, self.args.tgt_seq_length),
        )
        return generate_dummy_batch(num_tokens, self.collater, self.dict, src_len, tgt_len)

