import torch.nn as nn
from models.vivit.module.vivit import ViViT
from models.vivit.module.transformer import ClassificationHead


class MMViViT(nn.Module):
    def __init__(self, video_encoder, len_feature, num_classes, num_segments, fusion_type, pooling_type=""):
        super(MMViViT, self).__init__()
        # Fusion
        self.fusion_type = fusion_type
        self.num_classes = num_classes

        self.vivit = ViViT(
            num_classes=num_classes,
            num_frames=num_segments,
            pretrain_pth="models/vivit/weight/vit_base_patch16_224.pth",
            img_size=224,
            patch_size=16,
            embed_dims=len_feature,
            in_channels=3,
            num_heads=6,
            num_transformer_layers=1,
            conv_type='Conv3d',
            attention_type='fact_encoder',
            return_cls_token=True
        )

        self.cls_head = ClassificationHead(
            in_channels=768,
            num_classes=num_classes
        )


    def forward(self, visual_data, audio_data):
        batch_size, num_devices, num_segments, D, H, W = visual_data.size()

        visual_data = visual_data.view(-1, visual_data.size(2), visual_data.size(3), visual_data.size(4), visual_data.size(5))

        vivit_output = self.vivit(visual_data)
        x_cls = self.cls_head(vivit_output)

        x_frames_cls = None
        return x_cls, x_frames_cls