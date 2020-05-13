import torch
from torch import nn
from torchvision.models.resnet import resnet18


class View(nn.Module):
    def __init__(self, output_shape=None):
        super(View, self).__init__()
        self.output_shape = output_shape

    def forward(self, x):
        return x.view(*self.output_shape)


class Backbone(nn.Module):
    def __init__(self,
                 backbone=resnet18(False),
                 embedding_size=512,
                 input_shape=(3, 224, 224),
                 is_seesaw_backbone=False):
        super(Backbone, self).__init__()

        self.backbone_embedding_size = nn.Sequential(
            *(list(backbone.children())[:-2 if is_seesaw_backbone else -1])
        )(torch.rand((1,) + input_shape)).view(-1).shape[0]

        # print('Backbone output size: {}'.format(self.backbone_embedding_size))

        self.backbone = nn.Sequential(
            *(list(backbone.children())[:-2 if is_seesaw_backbone else -1]),
            View((-1, self.backbone_embedding_size)),
            nn.Linear(self.backbone_embedding_size, embedding_size)
        )

    def forward(self, x):
       return self.backbone(x)


class ArcFaceLayer(nn.Module):
    """ArcFace layer"""
    def __init__(self,
                 embedding_size=512,
                 num_classes=1000,
                 scale=64.0,
                 margin=0.5,
                 **params):
        """
        Class constructor
        Args:
            embedding_size: layer embedding size
            num_classes: count of classes
        """
        super(ArcFaceLayer, self).__init__()

        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin
        self.W = nn.Linear(embedding_size, num_classes)

    def forward(self, x, y):
        x = nn.functional.normalize(
            x,
            p=2,
            dim=1
        )

        w = nn.functional.normalize(
            self.W.weight.view(-1, self.num_classes),
            p=2,
            dim=0
        )

        y = torch.nn.functional.one_hot(
            y,
            num_classes=self.num_classes
        ).to(x.dtype)
        y = y.view(-1, self.num_classes)

        logits = x @ w  # dot product

        # clip logits to prevent zero division when backward
        theta = torch.acos(logits.clamp(-1.0 + 1E-7, 1.0 - 1E-7))

        target_logits = torch.cos(theta + self.margin)

        logits = logits * (1 - y) + target_logits * y
        logits *= self.scale  # feature re-scale

        return logits


class FaceRecognitionModel(nn.Module):
    def __init__(self,
                 backbone,
                 head):
        super(FaceRecognitionModel, self).__init__()

        self.backbone = backbone
        self.head = head

    def forward(self, x, y):
        return self.head(self.backbone(x), y)

    def inference(self, x):
        return self.backbone(x)
