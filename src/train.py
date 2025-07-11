"""
Training script for Pet Mood Detector model.
"""
import os
import time
import argparse
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import make_model, PetMoodClassifier
from datamodule import get_loaders


def train_model(
    model: nn.Module,
    dataloaders: Dict,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: lr_scheduler._LRScheduler,
    writer: SummaryWriter,
    num_epochs: int = 25,
    device: str = "cuda",
    save_dir: str = "models"
) -> nn.Module:
    """
    Train the model.
    
    Args:
        model: The model to train
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        writer: TensorBoard writer
        num_epochs: Number of epochs to train for
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save model checkpoints
        
    Returns:
        Trained model
    """
    since = time.time()
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize best model tracking
    best_model_weights = model.state_dict()
    best_acc = 0.0
    
    # Track metrics
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data
            pbar = tqdm(dataloaders[phase], desc=f"{phase}")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                # Track history only in train phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({"loss": loss.item()})
            
            # Compute epoch metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # Update history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())
            
            # Log to tensorboard
            writer.add_scalar(f'Loss/{phase}', epoch_loss, epoch)
            writer.add_scalar(f'Accuracy/{phase}', epoch_acc, epoch)
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = model.state_dict()
                # Save best model
                torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
                print(f"Saved new best model with accuracy: {best_acc:.4f}")
        
        # Step the scheduler
        if scheduler:
            scheduler.step()
        
        # Save checkpoint every few epochs
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f"{save_dir}/checkpoint_epoch_{epoch + 1}.pth")
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    # Plot and save training curves
    plot_training_curves(history, save_dir)
    
    return model


def plot_training_curves(history: Dict[str, List[float]], save_dir: str) -> None:
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with training history
        save_dir: Directory to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png")
    plt.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Pet Mood Detector")
    parser.add_argument('--data_dir', type=str, default='data/master_folder',
                        help='Directory with train, valid, and test folders')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs to train for')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'mobilenet_v2'],
                        help='Backbone architecture')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of worker threads for data loading')
    parser.add_argument('--save_dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    args = parser.parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    dataloaders = get_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size
    )
    
    # Count number of classes
    num_classes = len(dataloaders['train'].dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {dataloaders['train'].dataset.classes}")
    
    # Create the model
    model = PetMoodClassifier(
        num_classes=num_classes,
        backbone=args.backbone,
        pretrained=True
    ).to(device)
    
    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Step LR every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Create TensorBoard writer
    log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
    
    # Train the model
    trained_model = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        num_epochs=args.num_epochs,
        device=device,
        save_dir=args.save_dir
    )
    
    # Save the final model
    trained_model.save(f"{args.save_dir}/final_model.pth")
    print(f"Final model saved to {args.save_dir}/final_model.pth")
    
    # Close TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
