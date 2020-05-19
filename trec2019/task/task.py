from torch import nn
from torch.nn import functional as F


class ClassificationTask(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        # init
        self._init_layers()

    def _init_layers(self):
        input_size = self.hparams.model.n[-1]
        hidden_size = self.hparams.task.hidden_size
        output_size = self.hparams.task.output_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, x, target):
        # assert len(x.shape) == 2  # (minibatch, C)
        return self.layers(x)

    # Task Loss: Classification
    def loss(self, logit, target):
        # x.shape() == (minibatch, C)
        return F.cross_entropy(logit, target)


class RankingTask(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def _init_layers(self):
        pass

    # Task Loss: Ranking
    def distance(self, x1, x2):
        # TODO: 고민 필요
        # return torch.pow(a - b, 2).sum(1).sqrt()
        # return F.cosine_similarity(a, b)
        return torch.norm(x1 - x2, dim=1)

    def forward(self):
        pass

    def loss(self, q, pos, neg):
        distance_p = self.distance(q, pos)
        distance_n = self.distance(q, neg)
        # Should be distance_n > distance_p, so mark all as 1 (not -1)
        return F.margin_ranking_loss(
            distance_n, distance_p, torch.ones_like(distance_p), margin=1.0
        )
