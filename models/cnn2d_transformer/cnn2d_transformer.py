import torch
import torch.nn as nn
from models.cnn2d.resnet_2d import ResNetFeatureExtractor
from models.cnn2d.audio_extraction.ast import SimpleASTModel
from models.cnn2d_transformer.temporal_transformer.temporal_transformer import TemporalTransformer

class CNN2D_Transformer(nn.Module):
    def __init__(self, video_encoder, len_feature, num_classes, num_segments, fusion_type, pooling_type=""):
        super(CNN2D_Transformer, self).__init__()
        # Fusion
        self.fusion_type = fusion_type
        self.len_feature = len_feature + 384 # 384 is the output dimension of the audio encoder
        self.num_classes = num_classes

        # Spatial Encoder
        self.video_encoder = ResNetFeatureExtractor(video_encoder, image_pretrained=True, image_trainable=True)
        # Audio Encoder
        self.audio_encoder = SimpleASTModel(output_dim=384)

        # Temporal Transformer
        self.temporal_transformer = TemporalTransformer(
            num_classes=num_classes,
            num_frames=num_segments,
            img_size=224,
            patch_size=16,
            embed_dims=self.len_feature,
            in_channels=3,
            num_heads=12,
            tube_size=1,
            dropout_p = 0.1,
            num_transformer_layers=1,
            num_time_transformer_layers=2,
            conv_type='Conv3d',
            attention_type='temporal_only',
            return_cls_token=True,
            use_learnable_pos_emb=False,
        )

        # Device Interaction Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.len_feature, nhead=4, dim_feedforward=256, dropout=0.1)
        self.device_transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)


        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.len_feature),
            nn.Linear(self.len_feature, self.num_classes)
        )

        self.mlp_head_frames = nn.Conv1d(in_channels=self.len_feature, out_channels=self.num_classes, kernel_size=1, padding=0)

        self.mlp_human_frames = nn.Conv1d(in_channels=384, out_channels=1, kernel_size=1, padding=0)


    def forward(self, x, audio_data, human_info):
        """
        Input:
            x: (batch_size, num_devices, num_segments, D, H, W)
        """
        batch_size, num_devices, num_segments, D, H, W = x.size()

        ### Spatial Encoder -> Features
        x_video_feature = x.view(-1, x.size(3), x.size(4), x.size(5))
        x_video_feature = self.video_encoder(x_video_feature)
        x_video_feature = x_video_feature.view(batch_size, num_devices, num_segments, -1)

        ### Get human_frames
        x_human_frames = self.mlp_human_frames(x_video_feature.view(batch_size * num_devices, num_segments, -1).permute(0, 2, 1)).permute(0, 2, 1)
        ###

        ### Audio Encoder -> Features
        x_audio_feature = audio_data.view(-1, audio_data.size(3), audio_data.size(4))
        x_audio_feature = self.audio_encoder(x_audio_feature)
        x_audio_feature = x_audio_feature.view(batch_size, num_devices, num_segments, -1)

        # Fusion of audio and video features
        x_video_feature = torch.cat((x_video_feature, x_audio_feature), dim=-1) # (batch_size, num_devices, num_segments, len_feature)

        # Reshape and pass through Temporal Transformer
        x_video_feature = x_video_feature.view(batch_size * num_devices, num_segments, -1)
        x_cls, x_features = self.temporal_transformer(x_video_feature)
        
        ### Frames 
        x_cls_frames = self.mlp_head_frames(x_features.permute(0, 2, 1)).permute(0, 2, 1)
        x_cls_frames = x_cls_frames.view(batch_size, num_devices, num_segments, -1)
        x_cls_frames, _ = torch.max(x_cls_frames, dim=1)
        
        # Device Relationship Transformer
        x_device = x_features.view(batch_size, num_devices, num_segments, -1)
        x_device = x_device.permute(0, 2, 1, 3).contiguous()  # Shape: (batch_size, num_segments, num_devices, features)
        x_device = x_device.view(batch_size * num_segments, num_devices, -1)
        x_device = self.device_transformer(x_device)  
        x_device = x_device.view(batch_size, num_segments, num_devices, -1).mean(dim=2)

        x_cls_final = self.mlp_head(x_device.mean(dim=1))

        return x_cls_final, x_cls_frames, None, x_human_frames 

        
        ### way 3
        x_cls = x_features.mean(dim=1)

        ### x_cls: way 2
        x_cls = self.mlp_head(x_cls)
        x_cls = x_cls.view(batch_size, num_devices, -1)
        x_cls, _ = torch.max(x_cls, dim=1)

        # x_cls = x_temporal_cls.view(batch_size, num_devices, -1)
        # x_cls, _ = torch.max(x_cls, dim=1)
        # x_cls = self.mlp_head(x_cls)

        return x_cls, x_cls_frames, None, None

        import pdb; pdb.set_trace()

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