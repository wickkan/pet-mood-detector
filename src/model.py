"""
Model definition for Pet Mood Detector.
"""
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torchvision import models


def make_model(backbone="resnet18", num_classes=4, pretrained=True, lightweight=False):
    """
    Create a model with the specified backbone.
    
    Args:
        backbone (str): The backbone to use ('resnet18', 'mobilenet_v2', or 'mobilenet_v3_small')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
        lightweight (bool): If True, use a smaller model architecture
        
    Returns:
        model: The created model
    """
    # Add SSL verification bypass for downloading pretrained models
    import os
    import ssl
    if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

    if backbone == "resnet18":
        # Use ResNet18 as backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = PetMoodClassifier(backbone=backbone, num_classes=num_classes, weights=weights, lightweight=lightweight)
    elif backbone == "mobilenet_v2":
        # Use MobileNetV2 as backbone
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = PetMoodClassifier(backbone=backbone, num_classes=num_classes, weights=weights, lightweight=lightweight)
    elif backbone == "mobilenet_v3_small":
        # Use MobileNetV3 Small as backbone (much smaller model)
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = PetMoodClassifier(backbone=backbone, num_classes=num_classes, weights=weights, lightweight=lightweight)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model


class PetMoodClassifier(nn.Module):
    """
    Pet mood classifier model using a pre-trained backbone.
    """
    
    def __init__(
        self,
        num_classes: int = 4,
        backbone: str = "resnet18",
        pretrained: bool = True,
        dropout_rate: float = 0.2,
        lightweight: bool = False,
        weights=None
    ):
        """
        Initialize the pet mood classifier.
        
        Args:
            num_classes: Number of mood classes to predict
            backbone: Base model architecture ('resnet18', 'mobilenet_v2', or 'mobilenet_v3_small')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout probability for regularization
            lightweight: If True, use a smaller model architecture
            weights: Weights to use for the model
        """
        super(PetMoodClassifier, self).__init__()
        
        self.backbone = backbone
        self.num_classes = num_classes
        
        # Create base model
        if backbone == "resnet18":
            base_model = models.resnet18(weights=weights)
            self.feature_extractor = nn.Sequential(*list(base_model.children())[:-1])
            self.feature_dim = 512
        elif backbone == "mobilenet_v2":
            base_model = models.mobilenet_v2(weights=weights)
            self.feature_extractor = base_model.features
            self.feature_dim = base_model.last_channel
        elif backbone == "mobilenet_v3_small":
            base_model = models.mobilenet_v3_small(weights=weights)
            self.feature_extractor = base_model.features
            self.feature_dim = base_model.classifier[0].in_features
        
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Classification head
        if lightweight:
            # Smaller classification head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, num_classes)
            )
        else:
            # Standard classification head
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(self.feature_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(256, num_classes)
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
