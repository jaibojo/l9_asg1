import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tabulate import tabulate
import time
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid

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

def visualize_dataset_samples(dataset, num_images=16, title="Sample Images"):
    """
    Visualize sample images from the dataset
    Args:
        dataset: PyTorch dataset
        num_images: Number of images to display
        title: Title for the plot
    """
    # Get a random sample of images
    indices = torch.randperm(len(dataset))[:num_images]
    
    # Create figure
    fig = plt.figure(figsize=(15, 15))
    plt.title(title)
    
    # Get class names if available
    class_names = dataset.classes if hasattr(dataset, 'classes') else None
    
    for i, idx in enumerate(indices):
        img, label = dataset[idx]
        
        # Convert tensor to image
        img = img.numpy().transpose((1, 2, 0))
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        # Add subplot
        plt.subplot(int(np.sqrt(num_images)), int(np.sqrt(num_images)), i + 1)
        plt.imshow(img)
        
        # Add label
        if class_names:
            plt.title(f"Class: {class_names[label]}")
        else:
            plt.title(f"Label: {label}")
            
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def get_dataloaders(train_path, val_path, batch_size=256, num_workers=4, use_imagenet=False, imagenet_path=None, val_split=0.1):
    """
    Create training and validation dataloaders
    Args:
        train_path: Path to training data directory
        val_path: Path to validation data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        use_imagenet: Whether to use ImageNet dataset
        imagenet_path: Path to ImageNet dataset root directory
        val_split: Fraction of data to use for validation (default: 0.1)
    """
    transform_train, transform_val = get_transforms()
    
    if use_imagenet:
        if not imagenet_path:
            raise ValueError(
                "ImageNet path must be provided when using ImageNet dataset. "
                "Provide the path to the ImageNet directory using --imagenet-path argument."
            )
        try:
            print(f"\nLoading ImageNet dataset from: {imagenet_path}")
            print("This might take a few moments...")
            
            # Load the full dataset
            full_dataset = torchvision.datasets.ImageFolder(
                root=imagenet_path,
                transform=None  # We'll apply transforms after splitting
            )
            
            # Calculate split sizes
            total_size = len(full_dataset)
            val_size = int(total_size * val_split)
            train_size = total_size - val_size
            
            # Create train/val splits
            train_dataset, val_dataset = random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
            # Create new datasets with appropriate transforms
            train_dataset = TransformedSubset(train_dataset, transform_train)
            val_dataset = TransformedSubset(val_dataset, transform_val)
            
            # Print dataset statistics
            print(f"\nDataset split complete:")
            print(f"Total images: {total_size}")
            print(f"Training images: {len(train_dataset)}")
            print(f"Validation images: {len(val_dataset)}")
            print(f"Number of classes: {len(full_dataset.classes)}")
            
            print("\nSample class names:")
            for i, class_name in enumerate(full_dataset.classes[:5]):
                print(f"  {i}: {class_name}")
            
            # Visualize some training samples
            print("\nVisualizing some training samples...")
            visualize_dataset_samples(train_dataset, title="Training Samples")
            
            print("\nVisualizing some validation samples...")
            visualize_dataset_samples(val_dataset, title="Validation Samples")
                
        except Exception as e:
            raise RuntimeError(
                f"Error loading ImageNet dataset from {imagenet_path}. "
                f"Original error: {str(e)}"
            ) from e
    else:
        train_dataset = torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)
        val_dataset = torchvision.datasets.ImageFolder(root=val_path, transform=transform_val)
        
        # Visualize some training samples
        print("\nVisualizing some training samples...")
        visualize_dataset_samples(train_dataset, title="Training Samples")
        
        print("\nVisualizing some validation samples...")
        visualize_dataset_samples(val_dataset, title="Validation Samples")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # This helps speed up data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
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

class TransformedSubset:
    """
    Wrapper for applying transforms to a subset of a dataset
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset) 