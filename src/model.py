"""
Model definition for Pet Mood Detector.
"""
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision import models


def make_model(
    num_classes: int = 4,  # 'angry', 'happy', 'sad', and 'other'
    backbone: str = "resnet18", 
    pretrained: bool = True
) -> nn.Module:
    """
    Create a model for pet mood detection.
    
    Args:
        num_classes: Number of mood classes to predict
        backbone: Base model architecture ('resnet18' or 'mobilenet_v2')
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model with the specified architecture
    """
    if backbone == "resnet18":
        # Use ResNet18 as backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        # Replace the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif backbone == "mobilenet_v2":
        # Use MobileNetV2 as backbone
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        # Replace the final classifier
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    
    return model


class PetMoodClassifier(nn.Module):
    """
    Complete classifier for pet mood detection with custom architecture options.
    """
    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.2
    ):
        """
        Initialize the pet mood classifier.
        
        Args:
            num_classes: Number of mood classes to predict
            backbone: Base model architecture ('resnet18' or 'mobilenet_v2')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout probability for regularization
        """
        super(PetMoodClassifier, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Create base model
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 512
        
        elif backbone == "mobilenet_v2":
            weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
            base_model = models.mobilenet_v2(weights=weights)
            self.feature_extractor = base_model.features
            self.feature_dim = base_model.last_channel
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.feature_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Extract features
        features = self.feature_extractor(x)
        
        # Global average pooling
        features = features.mean([2, 3]) if len(features.shape) == 4 else features
        
        # Classification
        output = self.classifier(features)
        return output
    
    def save(self, path: str) -> None:
        """Save model state to a file."""
        torch.save({
            'state_dict': self.state_dict(),
            'backbone': self.backbone,
            'num_classes': self.num_classes
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'PetMoodClassifier':
        """Load a model from a saved state."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            num_classes=checkpoint['num_classes'],
            backbone=checkpoint['backbone']
        )
        model.load_state_dict(checkpoint['state_dict'])
        return model


if __name__ == "__main__":
    # Test the model
    model = make_model(num_classes=4)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Model output shape: {output.shape}")
    
    # Test the custom classifier
    classifier = PetMoodClassifier(num_classes=4)
    output = classifier(x)
    print(f"Classifier output shape: {output.shape}")
