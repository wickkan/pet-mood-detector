"""
Data module for Pet Mood Detector project.
"""
import os
from typing import Dict, Tuple, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


def get_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    """
    Get image transformations for training and validation/testing.
    
    Args:
        image_size: Target image size for resizing
        
    Returns:
        Dictionary with 'train' and 'val' transform compositions
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    return data_transforms


def get_loaders(
    data_dir: str = 'data/master_folder',
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Dict[str, DataLoader]:
    """
    Create data loaders for training, validation, and test datasets.
    
    Args:
        data_dir: Root directory containing train, valid, and test folders
        batch_size: Batch size for the data loaders
        num_workers: Number of worker threads for data loading
        image_size: Target image size for resizing
    
    Returns:
        Dictionary with 'train', 'val', and 'test' data loaders
    """
    data_transforms = get_transforms(image_size)
    
    # Create dataset instances
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), 
                                      transform=data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'valid'), 
                                   transform=data_transforms['val']),
        'test': datasets.ImageFolder(os.path.join(data_dir, 'test'), 
                                    transform=data_transforms['val'])
    }
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size,
                          shuffle=True, num_workers=num_workers),
        'val': DataLoader(image_datasets['val'], batch_size=batch_size,
                        shuffle=False, num_workers=num_workers),
        'test': DataLoader(image_datasets['test'], batch_size=batch_size,
                         shuffle=False, num_workers=num_workers)
    }
    
    # Store the class names and sizes for reference
    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    
    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")
    
    return dataloaders


class PetMoodDataset(Dataset):
    """
    Custom dataset for pet mood detection with additional functionality.
    """
    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Args:
            root_dir: Directory with all the images organized in class folders
            transform: Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.class_to_idx = self.dataset.class_to_idx
        self.samples = self.dataset.samples
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]
    
    def get_labels(self) -> List[int]:
        """Get all labels in the dataset."""
        return [label for _, label in self.samples]


if __name__ == "__main__":
    # Test the data module
    dataloaders = get_loaders()
    
    # Print a sample batch
    for images, labels in dataloaders['train']:
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        break
