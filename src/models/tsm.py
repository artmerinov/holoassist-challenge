# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu
# Edited by artmerinov

import torch
import torch.nn as nn


class TSM(nn.Module):
    def __init__(self, num_segments: int, n_div: int):
        super(TSM, self).__init__()
        self.num_segments = num_segments
        self.fold_div = n_div

    def forward(self, x):
        n_t, c, h, w = x.size()
        t = self.num_segments
        n = n_t // t

        x = x.view(n, t, c, h, w)

        fold = c // self.fold_div

        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

        return out.view(n_t, c, h, w)
