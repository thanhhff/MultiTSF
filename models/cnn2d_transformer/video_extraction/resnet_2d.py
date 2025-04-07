import torch
from torch import nn
import timm

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="resnet10", image_pretrained=True, image_trainable=True):
        super(ResNetFeatureExtractor, self).__init__()
        self.model = timm.create_model(model_name, pretrained=image_pretrained, num_classes=0)
        for param in self.model.parameters():
            param.requires_grad = image_trainable

    def forward(self, x):
        x = self.model(x)
        return x
    

### Test class
def test():
    model = ResNetFeatureExtractor(image_pretrained=True, image_trainable=True)
    x = torch.randn(10, 3, 224, 224)
    print(model(x).shape)
