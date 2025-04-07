from torch import nn
from transformers import ASTConfig, ASTForAudioClassification

class ASTFeatureExtractor(nn.Module):
    def __init__(self, output_dim=512):
        super(ASTFeatureExtractor, self).__init__()

        modified_config = ASTConfig(num_hidden_layers=1, num_labels=output_dim)
        self.model = ASTForAudioClassification(modified_config)

    def forward(self, x):
        x = self.model(x)
        return x.logit
    

class SimpleASTModel(nn.Module):
    def __init__(self, output_dim=512):
        super(SimpleASTModel, self).__init__()
        # 1. Fully Connected Layer 1
        self.fc1 = nn.Linear(1024 * 128, 512)
        self.relu1 = nn.ReLU()

        # 2. Fully Connected Layer 2
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()

        # 3. Fully Connected Layer 3
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()

        # 4. Output Layer
        self.output = nn.Linear(128, output_dim)

    def forward(self, x):
        # Flatten the input from (batch_size, 1024, 128) to (batch_size, 1024 * 128)
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)

        # Output layer
        logits = self.output(x)
        
        return logits