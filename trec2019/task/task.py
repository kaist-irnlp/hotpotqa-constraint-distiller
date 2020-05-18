from torch import nn


class ClassificationTask(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt


class RankingTask(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt


# Task Loss: Ranking
def distance(self, x1, x2):
    # TODO: 고민 필요
    # return torch.pow(a - b, 2).sum(1).sqrt()
    # return F.cosine_similarity(a, b)
    return torch.norm(x1 - x2, dim=1)


def loss_triplet(self, q, pos, neg):
    distance_p = self.distance(q, pos)
    distance_n = self.distance(q, neg)
    # Should be distance_n > distance_p, so mark all as 1 (not -1)
    return F.margin_ranking_loss(
        distance_n, distance_p, torch.ones_like(distance_p), margin=1.0
    )


# Task Loss: Classification
def loss_classify(self, input, target):
    # input.shape() == (minibatch, C)
    return F.cross_entropy(input, target)
