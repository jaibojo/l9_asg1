# ResNet-50 Training

This repository contains code for training a ResNet-50 model using PyTorch. It supports both a small test dataset and the full ImageNet-1k dataset.

## Features

- ResNet-50 architecture implementation
- OneCycle learning rate policy
- Comprehensive logging system (console, file, and markdown)
- Automatic model checkpointing
- Progress tracking with detailed metrics
- Support for both test and full datasets
- Memory-efficient training with gradient accumulation
- ImageNet-1k dataset support with automatic downloading

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jaibojo/l9_asg1.git
   cd l9_asg1
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

- `model.py` - ResNet-50 model architecture implementation
- `train.py` - Main training script with training loop and configuration
- `utils.py` - Utility functions for logging, metrics, and data handling
- `setup_test_data.py` - Script to set up a small test dataset
- `requirements.txt` - Python package dependencies
- `checkpoints/` - Directory for saved model checkpoints
- `test_data/` - Directory containing test dataset
- `TRAINING_LOG.md` - Detailed training progress and metrics log

## Usage

### Training with Test Dataset

1. Set up the test dataset:
   ```bash
   python setup_test_data.py
   ```

2. Start training (default 100 epochs):
   ```bash
   python train.py
   ```

### Training with ImageNet

To train on ImageNet-1k dataset:

1. Download the ImageNet dataset from Kaggle:
   - Go to [Kaggle ImageNet Dataset](https://www.kaggle.com/c/imagenet-object-localization-challenge/data)
   - If you haven't already, install the Kaggle CLI:
     ```bash
     pip install kaggle
     ```
   - Configure your Kaggle credentials:
     - Go to your Kaggle account settings (https://www.kaggle.com/account)
     - Click "Create New API Token" to download `kaggle.json`
     - Place the downloaded `kaggle.json` in `~/.kaggle/` directory
     - Set proper permissions:
       ```bash
       mkdir -p ~/.kaggle
       mv kaggle.json ~/.kaggle/
       chmod 600 ~/.kaggle/kaggle.json
       ```
   - Download the dataset:
     ```bash
     # Create directory for ImageNet
     mkdir imagenet
     cd imagenet
     
     # Download the dataset
     kaggle competitions download -c imagenet-object-localization-challenge
     
     # Extract the files
     unzip imagenet-object-localization-challenge.zip
     ```

2. Start training:
   ```bash
   python train.py --use-imagenet --imagenet-path /path/to/imagenet/ILSVRC/Data/CLS-LOC --epochs 100
   ```

Note: The ImageNet dataset is large (~150GB). Make sure you have sufficient disk space before downloading.

### Command Line Arguments

- `--train-path`: Path to training data (default: 'test_data/train')
- `--val-path`: Path to validation data (default: 'test_data/val')
- `--checkpoint-dir`: Directory to save checkpoints (default: 'checkpoints')
- `--batch-size`: Batch size (default: 32)
- `--num-workers`: Number of data loading workers (default: 4)
- `--learning-rate`: Initial learning rate (default: 0.1)
- `--momentum`: SGD momentum (default: 0.9)
- `--weight-decay`: Weight decay (default: 1e-4)
- `--epochs`: Number of epochs to train (default: 100)
- `--pretrained`: Use pretrained model (flag)
- `--use-imagenet`: Use ImageNet dataset (flag)
- `--imagenet-path`: Path to ImageNet dataset root directory (required when using --use-imagenet)

## Requirements

- Python 3.6+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- Other dependencies as listed in requirements.txt

## Monitoring

Training progress can be monitored through:
- Console output with real-time metrics
- TRAINING_LOG.md for detailed progress tracking
- Model checkpoints saved periodically

## License

[Specify License]