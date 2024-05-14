# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
# Edited

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class FocalLoss(nn.Module):
    """
    Focal loss implementation.
    
    Args:
        gamma: The focusing parameter. Higher values of gamma increase
            the penalty for misclassified examples with low predicted probabilities.
            gamma >= 0.
        alpha: A vector of weights assigned to each class in the dataset.
            Alpha value closer to 1 increases the weight of minority classes,
            helping in class imbalance scenarios.
            0 <= alpha <= 1.
        size_average: By default, the losses are averaged over observations for each minibatch. 
            However, if the field size_average is set to False, the losses are instead summed 
            for each minibatch. 
    """
    def __init__(self, 
                 gamma: float = 0.0, 
                 alpha: Optional[List[float]] = None,
                 size_average: Optional[bool] = True
        ) -> None:
        super(FocalLoss, self).__init__()

        # default parameters
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

        # If alpha is specified, convert it to a tensor
        if self.alpha is not None:
            self.alpha = torch.Tensor(self.alpha)


    def forward(self, x, target):
        """ 
        Get focal loss.

        Args:
            x (Variable):  the prediction with shape [batch_size, number of classes]
            target (Variable): the answer with shape [batch_size, number of classes]

        Returns:
            Variable (float): loss
        """
        if x.dim() > 2:
            x = x.view(x.size(0),x.size(1),-1)  # N,C,H,W => N,C,H*W
            x = x.transpose(1,2)    # N,C,H*W => N,H*W,C
            x = x.contiguous().view(-1,x.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(x, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.data.exp()

        if self.alpha is not None:
            if self.alpha.type() != x.data.type():
                self.alpha = self.alpha.type_as(x.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt

        if self.size_average: 
            return loss.mean()
        else:
            return loss.sum()