import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.ce_criterion = nn.BCELoss()

    def forward(self, logits, label):
        label = label / torch.sum(label, dim=1, keepdim=True) + 1e-10
        loss = -torch.mean(torch.sum(label * F.log_softmax(logits, dim=1), dim=1), dim=0)
        return loss


class GeneralizedCE(nn.Module):
    def __init__(self, q):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, label):
        assert logits.shape[0] == label.shape[0]
        assert logits.shape[1] == label.shape[1]
        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7
        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label), dim=1)/neg_factor)
        return first_term + second_term


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Binary version of the focal loss.
        alpha: balancing factor.
        gamma: modulating factor.
        
        Args:
            alpha (float): Weighting factor for the positive class (default: 0.25).
            gamma (float): Focusing parameter to adjust the rate at which easy
                           examples are down-weighted (default: 2.0).
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Apply focal loss.
        
        Args:
            inputs: Logits from the model's output (before sigmoid).
            targets: Ground truth labels (same shape as inputs).
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.float()
        # Here we calculate the loss
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
        

nINF = -100
class TwoWayLoss(nn.Module):
    def __init__(self, Tp=4., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        # Calculate hard positive/negative logits
        pmask = y.masked_fill(y <= 0, nINF).masked_fill(y > 0, float(0.0))
        plogit_class = torch.logsumexp(-x/self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x/self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]
    
        nmask = y.masked_fill(y != 0, nINF).masked_fill(y == 0, float(0.0))
        nlogit_class = torch.logsumexp(x/self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x/self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        return torch.nn.functional.softplus(nlogit_class + plogit_class).mean() + \
                torch.nn.functional.softplus(nlogit_sample + plogit_sample).mean()