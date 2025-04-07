import torch
from torch import nn
from transformers import VivitConfig, VivitModel


class MMViViT(nn.Module):
    def __init__(self, video_encoder, len_feature, num_classes, num_segments, fusion_type, pooling_type=""):
        super().__init__()
        self.fusion_type = fusion_type
        config = VivitConfig.from_pretrained("google/vivit-b-16x2-kinetics400", num_frames=num_segments)
        config.num_hidden_layers = 1
        self.backbone = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=config, ignore_mismatched_sizes=True)

        self.classif = nn.Linear(in_features=self.backbone.config.hidden_size, out_features=num_classes)

    def forward(self, images, is_training=True):
        b, d, t, c, h, w = images.size()
        new_images = images.reshape(-1, t, c, h, w)

        x = self.backbone(new_images)[0]
        x = x[:, 0]

        x = x.reshape(b, d, -1)
        if self.fusion_type == 'max':
            x, _ = x.max(dim=1)
        x = self.classif(x)

        return x, None