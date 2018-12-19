"""
The implementation of GHM-C and GHM-R losses.
Details can be found in the paper `Gradient Harmonized Single-stage Detector`:
https://arxiv.org/abs/1811.05181

Copyright (c) 2018 Multimedia Laboratory, CUHK.
Licensed under the MIT License (see LICENSE for details)
Written by Buyu Li
"""

import torch
import torch.nn.functional as F


class GHMC_Loss:
    def __init__(self, bins=10, momentum=0):
        self.bins = bins
        self.momentum = momentum
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] += 1e-6
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target, mask):
        """ Args:
        input [batch_num, class_num]:
            The direct prediction of classification fc layer.
        target [batch_num, class_num]:
            Binary target (0 or 1) for each sample each class. The value is -1
            when the sample is ignored.
        """
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(input)

        # gradient length
        g = torch.abs(input.sigmoid().detach() - target)

        valid = mask > 0
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            input, target, weights, reduction='sum') / tot
        return loss


class GHMR_Loss:
    def __init__(self, mu=0.02, bins=10, momentum=0):
        self.mu = mu
        self.bins = bins
        self.edges = [float(x) / bins for x in range(bins+1)]
        self.edges[-1] = 1e3
        self.momentum = momentum
        if momentum > 0:
            self.acc_sum = [0.0 for _ in range(bins)]

    def calc(self, input, target, mask):
        """ Args:
        input [batch_num, 4 (* class_num)]:
            The prediction of box regression layer. Channel number can be 4 or
            (4 * class_num) depending on whether it is class-agnostic.
        target [batch_num, 4 (* class_num)]:
            The target regression values with the same size of input.
        """
        mu = self.mu
        edges = self.edges
        mmt = self.momentum

        # ASL1 loss
        diff = input - target
        loss = torch.sqrt(diff * diff + mu * mu) - mu

        # gradient length
        g = torch.abs(diff / torch.sqrt(mu * mu + diff * diff)).detach()
        weights = torch.zeros_like(g)

        valid = mask > 0
        tot = max(mask.float().sum().item(), 1.0)
        n = 0  # n: valid bins
        for i in range(self.bins):
            inds = (g >= edges[i]) & (g < edges[i+1]) & valid
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                n += 1
                if mmt > 0:
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
        if n > 0:
            weights /= n

        loss = loss * weights
        loss = loss.sum() / tot
        return loss
