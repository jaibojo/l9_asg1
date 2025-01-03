import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import get_model
from utils import get_dataloaders, validate, MetricLogger

def train(config):
    # Create directories for checkpoints and logs
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    writer = SummaryWriter(config['log_dir'])

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Get dataloaders
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
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', current_lr, epoch)
        
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
    writer.close()

if __name__ == '__main__':
    config = {
        'train_path': 'path/to/train',
        'val_path': 'path/to/val',
        'checkpoint_dir': 'checkpoints',
        'log_dir': 'runs/resnet50_training',
        'num_classes': 1000,
        'batch_size': 256,
        'num_workers': 4,
        'learning_rate': 0.1,
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'epochs': 100,
        'pretrained': False
    }
    
    train(config) 