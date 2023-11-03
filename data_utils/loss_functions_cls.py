import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class CombinedMetricDiffCE(nn.Module):
    def __init__(self, vector_kernel="cosine_distance", alpha=0.5, smoothing=0.1, reduction="mean", device=torch.device('cuda')):
        super(CombinedMetricDiffCE, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
        self.device = device
        self.sce = LabelSmoothingCrossEntropy(smoothing=smoothing, reduction=reduction)

        dir = torch.tensor([ 
            [0.0000, 0.0000, 1.0000],                       # 0
            [0.0000, 0.0000, -1.0000],                      # 1
            
            [0.0000, -0.7071, 0.7071],                      # 2
            [0.0000, -1.0000, 0.0000],                      # 3
            [0.0000, -0.7071, -0.7071],                     # 4
            [0.0000, 0.7071, -0.7071],                      # 5
            [0.0000, 1.0000, 0.0000],                       # 6
            [0.0000, 0.7071, 0.7071],                       # 7
                
            [0.7071, 0.0000, 0.7071],                       # 8
            [1.0000, 0.0000, 0.0000],                       # 9
            [0.7071, 0.0000, -0.7071],                      # 10
            [-0.7071, 0.0000, -0.7071],                     # 11
            [-1.0000, 0.0000, 0.0000],                      # 12
            [-0.7071, 0.0000, 0.7071],                      # 13
                
            [0.5000, -0.7071, 0.5000],                      # 14
            [-0.5000, -0.7071, -0.5000],                    # 15
            [-0.5000, 0.7071, -0.5000],                     # 16
            [0.5000, 0.7071, 0.5000],                       # 17
                
            [0.7071, -0.7071, 0.0000],                      # 18
            [-0.7071, -0.7071, 0.0000],                     # 19
            [-0.7071, 0.7071, 0.0000],                      # 20
            [0.7071, 0.7071, 0.0000],                       # 21
                
            [0.5000, -0.7071, -0.5000],                     # 22
            [-0.5000, -0.7071, 0.5000],                     # 23
            [-0.5000, 0.7071, 0.5000],                      # 24
            [0.5000, 0.7071, -0.5000]]).to(self.device)     # 25
        
        N, _ = dir.shape
        self.label_weights = torch.zeros((N,N)).to(self.device)
        for idx1 in range(N):
            for idx2 in range(N):         
                # Compute metric difference between vectors using a predetermined kernel (phi in the SORD paper) 
                # Only criteria is that a lower number represents being more "similar"
                if vector_kernel == "cosine_distance":
                    metric_diff = 1.0 - F.cosine_similarity(dir[idx1], dir[idx2], eps=1e-08, dim=-1)
                elif vector_kernel == "euclidean_distance":
                    metric_diff = (dir[idx1] - dir[idx2]).pow(2).sum().sqrt()
                elif vector_kernel == "manhattan_distance":
                    metric_diff = (dir[idx1] - dir[idx2]).abs().sum()

                self.label_weights[idx1, idx2] = metric_diff

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Softmax to get probability distribution
        probs = F.softmax(x, dim=1)

        pred_dir = torch.argmax(probs, dim=1)

        # Calculate Cosine Distance
        batch_size = len(pred_dir)

        dir_diff = torch.tensor([self.label_weights[pred_dir[idx], target[idx]] for idx in range(batch_size)]).to(self.device)

        dir_diff = torch.pow(dir_diff, 2)
        
        if self.reduction == "mean":
            dir_diff = dir_diff.mean()
        elif self.reduction == "sum":
            dir_diff = dir_diff.sum()

        # Calculate Cross Entropy Loss
        ce = self.sce(x, target)

        # Combine the two loss functions using a weighting factor
        loss = self.alpha * dir_diff + (1.0 - self.alpha) * ce
        return loss

class KLDivWrapper(nn.Module):
    def __init__(self, reduction="mean"):
        super(KLDivWrapper, self).__init__()

        if reduction == "mean":
            reduction = "batchmean"
        
        self.kldiv = nn.KLDivLoss(reduction=reduction, log_target=False)

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        klloss = self.kldiv(logprobs, target)

        return klloss

class LabelSmoothingCrossEntropy(nn.Module):
    """ 
    NLL loss with label smoothing.
    From: https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/loss/cross_entropy.py#L11
    """
    def __init__(self, smoothing=0.1, reduction="mean"):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.reduction = reduction
        self.smoothing = smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n_class = x.size(1)
        eps = self.smoothing / n_class
        one_hot = torch.zeros_like(x).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(x, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss

class SoftTargetCrossEntropy(nn.Module):
    """ 
    CE loss with Soft Labels.
    Adapted from: https://github.com/huggingface/pytorch-image-models/blob/v0.6.12/timm/loss/cross_entropy.py#L11
    """

    def __init__(self, reduction="mean"):
        super(SoftTargetCrossEntropy, self).__init__()
        self. reduction = reduction

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss


def select_loss(loss_function, smoothing = True, smoothing_val = 0.1, vector_kernel="cosine_distance", alpha=0.5, reduction="mean", device=torch.device('cuda')):

    # If using Pytorch 2.0 CE, LS, and Soft-CE can be merged by nn.CrossEntropyLoss(smoothing=smoothing_val, reduction=reduction)
    # Earlier version of Pytorch does not support smoothing nor soft labels
    if loss_function == "CE":
        return nn.CrossEntropyLoss(reduction=reduction)
    elif loss_function == "LS":
        return LabelSmoothingCrossEntropy(smoothing=smoothing_val, reduction=reduction)
    elif loss_function == "KL" or loss_function == "KL-SORD":
        return KLDivWrapper(reduction=reduction)
    elif loss_function == "Soft-CE" or loss_function == "Soft-CE-SORD":
        return SoftTargetCrossEntropy(reduction=reduction)
    elif loss_function == "Combined-DiffCE":
        return CombinedMetricDiffCE(vector_kernel=vector_kernel, alpha=alpha, smoothing=smoothing_val, reduction=reduction, device=device)