import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class CosineDistance(torch.nn.Module):
    def __init__(self):
        super(CosineDistance, self).__init__()

    def forward(self, pred, target):
        ''' Calculate cosine distance loss. '''
        return torch.mean(1.0 - F.cosine_similarity(pred, target, eps=1e-08, dim=-1))


class EuclideanDistance(torch.nn.Module):
    def __init__(self):
        super(EuclideanDistance, self).__init__()

    def forward(self, pred, target):
        ''' Calculate euclidean distance loss. ''' 
        return torch.mean((pred - target).pow(2).sum().sqrt())
    

class ManhattanDistance(torch.nn.Module):
    def __init__(self):
        super(ManhattanDistance, self).__init__()

    def forward(self, pred, target):
        ''' Calculate manhattan distance loss. '''
        return torch.mean((pred - target).abs().sum())


def select_loss(vector_kernel="cosine_distance"):
    if vector_kernel == "cosine_distance":
        return CosineDistance()
    elif vector_kernel == "euclidean_distance":
        return EuclideanDistance()
    elif vector_kernel == "manhattan_distance":
        return ManhattanDistance()