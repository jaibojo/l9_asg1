import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tabulate import tabulate
import time
from datetime import datetime, timedelta
import os

def get_transforms():
    """
    Get training and validation transforms
    """
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return transform_train, transform_val

def get_dataloaders(train_path, val_path, batch_size=256, num_workers=4, use_imagenet=False, imagenet_path=None):
    """
    Create training and validation dataloaders
    Args:
        train_path: Path to training data directory
        val_path: Path to validation data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        use_imagenet: Whether to use ImageNet dataset
        imagenet_path: Path to ImageNet dataset root directory (should point to CLS-LOC directory)
    """
    transform_train, transform_val = get_transforms()
    
    if use_imagenet:
        if not imagenet_path:
            raise ValueError(
                "ImageNet path must be provided when using ImageNet dataset. "
                "Download the dataset from Kaggle: https://www.kaggle.com/c/imagenet-object-localization-challenge/data "
                "and provide the path to the CLS-LOC directory using --imagenet-path argument."
            )
        try:
            # For Kaggle's ImageNet, we use ImageFolder as the directory structure is different
            train_dir = os.path.join(imagenet_path, 'train')
            val_dir = os.path.join(imagenet_path, 'val')
            
            if not os.path.exists(train_dir) or not os.path.exists(val_dir):
                raise RuntimeError(
                    f"ImageNet directory structure not found in {imagenet_path}. "
                    "Expected structure:\n"
                    f"{imagenet_path}/\n"
                    "├── train/\n"
                    "│   ├── n01440764/\n"
                    "│   ├── n01443537/\n"
                    "│   └── ...\n"
                    "└── val/\n"
                    "    ├── n01440764/\n"
                    "    ├── n01443537/\n"
                    "    └── ..."
                )
            
            train_dataset = torchvision.datasets.ImageFolder(
                root=train_dir,
                transform=transform_train
            )
            val_dataset = torchvision.datasets.ImageFolder(
                root=val_dir,
                transform=transform_val
            )
            
            # Verify we have the expected number of classes
            if len(train_dataset.classes) != 1000:
                raise RuntimeError(
                    f"Expected 1000 classes in ImageNet dataset, but found {len(train_dataset.classes)}. "
                    "Please make sure you're using the correct ImageNet directory."
                )
                
        except Exception as e:
            raise RuntimeError(
                f"Error loading ImageNet dataset from {imagenet_path}. "
                "Make sure you have downloaded and extracted the dataset from Kaggle correctly.\n"
                f"Original error: {str(e)}"
            ) from e
    else:
        train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
        val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                          shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def validate(model, val_loader, criterion, device):
    """
    Validate the model
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return val_loss / len(val_loader), 100. * correct / total 

class MetricLogger:
    def __init__(self, log_file=None, markdown_file=None):
        """
        Initialize MetricLogger
        Args:
            log_file (str): Path to save the log file
            markdown_file (str): Path to save the markdown log file
        """
        self.headers = ['Epoch', 'Train Loss', 'Val Loss', 'Val Acc', 'LR', 'Time']
        self.metrics = []
        self.log_file = log_file or f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        self.epoch_start_time = None
        
        # Write headers to file
        with open(self.log_file, 'w') as f:
            f.write(tabulate([], headers=self.headers, tablefmt='grid') + '\n')
        
        self.markdown_file = markdown_file or 'TRAINING_LOG.md'
        self.training_start_time = time.time()
        self.total_time = 0
        self.epoch_times = []
        
        # Initialize markdown file
        self._init_markdown_file()
    
    def _init_markdown_file(self):
        """Initialize the markdown file with headers"""
        with open(self.markdown_file, 'w') as f:
            f.write("# ResNet-50 Training Log\n\n")
            f.write(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Training Configuration\n")
            f.write("- Model: ResNet-50\n")
            f.write("- Dataset: ImageNet-1k\n")
            f.write("- Batch Size: 256\n")
            f.write("- Initial Learning Rate: 0.1\n")
            f.write("- Optimizer: SGD with momentum (0.9)\n")
            f.write("- Weight Decay: 1e-4\n")
            f.write("- Training Epochs: 100\n")
            f.write("- Learning Rate Schedule: OneCycleLR\n\n")
            f.write("## Training Progress\n\n")
            f.write("| Epoch | Train Loss | Val Loss | Val Acc | Learning Rate | Time |\n")
            f.write("|-------|------------|----------|---------|---------------|------|\n")
    
    def start_epoch(self):
        """Start timing an epoch"""
        self.epoch_start_time = time.time()
    
    def log_metrics(self, epoch, train_loss, val_loss, val_acc, learning_rate):
        """
        Log metrics for current epoch
        """
        epoch_time = time.time() - self.epoch_start_time
        metrics = [
            epoch + 1,
            f'{train_loss:.4f}',
            f'{val_loss:.4f}',
            f'{val_acc:.2f}%',
            f'{learning_rate:.6f}',
            f'{epoch_time:.2f}s'
        ]
        self.metrics.append(metrics)
        
        # Print current epoch metrics
        print('\n' + tabulate([metrics], headers=self.headers, tablefmt='grid'))
        
        # Save to file
        with open(self.log_file, 'a') as f:
            f.write(tabulate([metrics], tablefmt='grid', colalign=['right']*len(self.headers)) + '\n') 
        
        # Add epoch time to list
        self.epoch_times.append(epoch_time)
        
        # Update markdown file
        with open(self.markdown_file, 'a') as f:
            f.write(f"| {epoch+1:<5} | {train_loss:.4f} | {val_loss:.4f} | {val_acc:.2f}% | {learning_rate:.6f} | {epoch_time:.0f}s |\n")
    
    def finish_training(self, best_acc, final_train_loss, final_val_loss):
        """Add summary statistics to markdown file"""
        total_time = time.time() - self.training_start_time
        avg_epoch_time = sum(self.epoch_times) / len(self.epoch_times)
        
        with open(self.markdown_file, 'a') as f:
            f.write("\n## Best Model Performance\n")
            f.write(f"- Best Validation Accuracy: {best_acc:.2f}%\n")
            f.write(f"- Final Training Loss: {final_train_loss:.4f}\n")
            f.write(f"- Final Validation Loss: {final_val_loss:.4f}\n\n")
            f.write("## Training Summary\n")
            f.write(f"- Total Training Time: {str(timedelta(seconds=int(total_time)))}\n")
            f.write(f"- Average Epoch Time: {avg_epoch_time:.2f}s\n")
            try:
                import psutil
                memory_usage = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024
                f.write(f"- Peak Memory Usage: {memory_usage:.1f} GB\n")
            except ImportError:
                pass 