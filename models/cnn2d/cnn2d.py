import torch
import torch.nn as nn
from models.cnn2d.resnet_2d import ResNetFeatureExtractor
from models.cnn2d.audio_extraction.ast import SimpleASTModel


class CNN2D(nn.Module):
    def __init__(self, video_encoder, len_feature, num_classes, num_segments, fusion_type, pooling_type=""):
        super(CNN2D, self).__init__()
        # Fusion
        self.fusion_type = fusion_type
        self.len_feature = len_feature + 384 # 384 is the output dimension of the audio encoder
        self.num_classes = num_classes

        # Spatial Encoder
        self.video_encoder = ResNetFeatureExtractor(video_encoder, image_pretrained=True, image_trainable=True)

        # Audio Encoder
        self.audio_encoder = SimpleASTModel(output_dim=384)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.len_feature),
            nn.Linear(self.len_feature, self.num_classes)
        )

        self.mlp_head_frames = nn.Conv1d(in_channels=self.len_feature, out_channels=self.num_classes, kernel_size=1, padding=0)


    def forward(self, x, audio_data):
        """
        Input:
            x: (batch_size, num_devices, num_segments, D, H, W)
        """
        batch_size, num_devices, num_segments, D, H, W = x.size()

        ### Spatial Encoder -> Features
        x_video_feature = x.view(-1, x.size(3), x.size(4), x.size(5))
        x_video_feature = self.video_encoder(x_video_feature)
        x_video_feature = x_video_feature.view(batch_size, num_devices, num_segments, -1)

        ### Audio Encoder -> Features
        x_audio_feature = audio_data.view(-1, audio_data.size(3), audio_data.size(4))
        x_audio_feature = self.audio_encoder(x_audio_feature)
        x_audio_feature = x_audio_feature.view(batch_size, num_devices, num_segments, -1)

        # Concatenate the features
        x_video_feature = torch.cat((x_video_feature, x_audio_feature), dim=-1)

        x_video_feature_frames = self.mlp_head_frames(x_video_feature.max(dim=1)[0].permute(0, 2, 1)).permute(0, 2, 1)

        if self.fusion_type == "max":
            # x_video_feature, _ = torch.max(x_video_feature, dim=2)
            x_video_feature, _ = torch.max(x_video_feature, dim=1)
            # x_video_feature, _ = torch.max(x_video_feature, dim=1)
            x_video_feature = torch.mean(x_video_feature, dim=1)

        elif self.fusion_type == "mean":
            x_video_feature = torch.mean(x_video_feature, dim=2)
            x_video_feature = torch.mean(x_video_feature, dim=1)

        # MLP Head
        x_cls = self.mlp_head(x_video_feature)

        return x_cls, x_video_feature_frames