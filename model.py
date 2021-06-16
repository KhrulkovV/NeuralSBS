import torchvision
import torch
import torch.nn as nn
import numpy as np

def _chop_model(model, remove=1):
    """
    Removes the last layer from the model.
    """
    model = torch.nn.Sequential(*(list(model.children())[:-remove]))
    return model

def _J(dim):
    res = np.zeros((dim, dim), dtype=np.float32)
    res[:dim // 2, dim // 2:] = np.eye(dim // 2)
    res[dim // 2:, :dim // 2] = -np.eye(dim // 2)
    return res

class SkewSimilarity(nn.Module):
    def __init__(self, embedding_dim):
        super(SkewSimilarity, self).__init__()
        std = np.sqrt(2.0 / embedding_dim)
        self.J = nn.Parameter(std * torch.randn(embedding_dim, embedding_dim))
        # #self.J = torch.Tensor(_J(embedding_dim))
        # self.register_buffer("J", torch.from_numpy(_J(embedding_dim)))

    def forward(self, x, y):
        J_ = 0.5 * (self.J - self.J.transpose(1, 0))
        return torch.sum(torch.matmul(x, J_) * y, dim=-1, keepdim=True)

class L2Normalize(nn.Module):
    def __init__(self, eps=1e-5):
        super(L2Normalize, self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, dim=-1, keepdim=True) + self.eps)

class ScoreModel(nn.Module):
    def __init__(self,
                 backbone_model,
                 embedding_dim=512):
        super(ScoreModel, self).__init__()
        self.features = backbone_model
        # self.processing = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, embedding_dim)
        # )
        self.embedding_dim = embedding_dim
        self.norm = L2Normalize()
        self.similarity = SkewSimilarity(embedding_dim=embedding_dim)
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        B, C, H, W = x.shape[0], x.shape[2], x.shape[3], x.shape[4]
        inp = x.view(-1, C, H, W)

        f = self.features(inp).view(B * 2, -1)
        f = f.view(B, 2, -1
        f1, f2 = f[:, 0, :], f[:, 1, :]

        f1 = self.norm(f1)
        f2 = self.norm(f2)
        d = self.similarity(f1, f2)
        return d

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.shape[0]
        return x.view(B, -1)

def get_score_model(name, pretrained=True, path=None):
    if path is None:
        model = getattr(torchvision.models, name)(pretrained=pretrained)
    else:
        model = getattr(torchvision.models, name)(pretrained=False)
        model.load_state_dict(torch.load(path + '/{}.pth'.format(name)))
    if name in ['densenet121',
                'densenet161',
                'densenet169',
                'densenet201']:
        out_features = model.classifier.in_features
        model = nn.Sequential(_chop_model(model), torch.nn.AdaptiveAvgPool2d((1, 1)))
        score_model = ScoreModel(model, embedding_dim=out_features)

    elif name in ['resnet18',
                  'resnet34',
                  'resnet50',
                  'resnet101',
                  'resnet152',
                  'resnext101_32x8d',
                  'resnext50_32x4d']:

        out_features = model.fc.in_features
        model = _chop_model(model)
        score_model = ScoreModel(model, embedding_dim=out_features)
    elif name in ['vgg16', 'vgg19']:
        model = nn.Sequential(
            model.features,
            model.avgpool,
            Flatten(),
            nn.Dropout(0.75),
            nn.Linear(25088, 512)
        )
        score_model = ScoreModel(model, embedding_dim=512)

    elif name in ['inception_v3']:
        model.aux_logits = False
        model.fc = nn.Linear(2048, 2048)
        score_model = ScoreModel(model, embedding_dim=2048)
    else:
        raise ValueError('Model is not supported.')

    return score_model
