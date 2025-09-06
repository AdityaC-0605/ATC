"""
PyTorch Model Definitions
"""
import torch
import torch.nn as nn
import torchvision.models as models
from config import DEVICE

class CattleClassifier(nn.Module):
    """ResNet-based cattle breed classifier"""
    
    def __init__(self, num_classes, pretrained=True):
        super(CattleClassifier, self).__init__()
        
        # Use ResNet50 as backbone (more stable than EfficientNet)
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Replace the final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def create_model(num_classes):
    """Create and initialize the model"""
    model = CattleClassifier(num_classes)
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✅ Model created on {DEVICE}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model
