import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from pathlib import Path
import argparse
import torchvision.datasets as datasets

from model import get_model
from utils import get_dataloaders, validate, MetricLogger

def validate_paths(train_path, val_path, download_imagenet=False):
    """Validate that data paths exist"""
    train_path = Path(train_path)
    val_path = Path(val_path)
    
    if download_imagenet:
        return
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training data path does not exist: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Validation data path does not exist: {val_path}")
    
    # Check if paths contain data
    if not any(train_path.iterdir()):
        raise ValueError(f"Training data path is empty: {train_path}")
    if not any(val_path.iterdir()):
        raise ValueError(f"Validation data path is empty: {val_path}")

def train(config):
    # Validate data paths
    validate_paths(config['train_path'], config['val_path'], config['use_imagenet'])

    # Create directories for checkpoints and logs
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get dataloaders
    if config['use_imagenet']:
        print("Using ImageNet dataset. This will download the dataset if not already present...")
        train_loader, val_loader = get_dataloaders(
            None, None,
            config['batch_size'],
            config['num_workers'],
            use_imagenet=True
        )
    else:
        train_loader, val_loader = get_dataloaders(
            config['train_path'],
            config['val_path'],
            config['batch_size'],
            config['num_workers']
        )

    # Initialize model
    model = get_model(num_classes=config['num_classes'], pretrained=config['pretrained'])
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 
                         lr=config['learning_rate'],
                         momentum=config['momentum'],
                         weight_decay=config['weight_decay'])

    # Define scheduler
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        steps_per_epoch=len(train_loader),
        epochs=config['epochs']
    )

    # Initialize metric logger
    logger = MetricLogger(
        log_file=os.path.join(config['checkpoint_dir'], 'training_log.txt'),
        markdown_file=os.path.join(config['checkpoint_dir'], 'TRAINING_LOG.md')
    )

    # Training loop
    best_acc = 0
    for epoch in range(config['epochs']):
        logger.start_epoch()
        model.train()
        running_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        train_loss = running_loss / len(train_loader)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.log_metrics(epoch, train_loss, val_loss, val_acc, current_lr)
        
        # Save checkpoint if validation accuracy improves
        if val_acc > best_acc:
            print(f'Saving checkpoint... Accuracy improved from {best_acc:.2f} to {val_acc:.2f}')
            state = {
                'model': model.state_dict(),
                'acc': val_acc,
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }
            torch.save(state, os.path.join(config['checkpoint_dir'], 'best_model.pth'))
            best_acc = val_acc

    logger.finish_training(best_acc, train_loss, val_loss)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResNet-50 model')
    parser.add_argument('--train-path', default='test_data/train', help='Path to training data')
    parser.add_argument('--val-path', default='test_data/val', help='Path to validation data')
    parser.add_argument('--checkpoint-dir', default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--learning-rate', type=float, default=0.1, help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained model')
    parser.add_argument('--use-imagenet', action='store_true', help='Use ImageNet dataset')
    parser.add_argument('--imagenet-path', type=str, help='Path to ImageNet dataset root directory')
    
    args = parser.parse_args()
    
    if args.use_imagenet and not args.imagenet_path:
        raise ValueError("When using ImageNet (--use-imagenet), you must specify --imagenet-path. "
                       "You need to manually download the ImageNet dataset from https://image-net.org/download-images.php "
                       "and provide the path to the downloaded dataset.")
    
    config = {
        'train_path': args.train_path,
        'val_path': args.val_path,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': 'runs/resnet50_training',
        'num_classes': 1000 if args.use_imagenet else 2,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'pretrained': args.pretrained,
        'use_imagenet': args.use_imagenet,
        'imagenet_path': args.imagenet_path if args.use_imagenet else None
    }
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    train(config) 