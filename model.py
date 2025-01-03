import torch
import torchvision
import torch.nn as nn

def get_model(num_classes=1000, pretrained=False):
    """
    Create and initialize the ResNet-50 model
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use pretrained weights
    """
    model = torchvision.models.resnet50(pretrained=pretrained)
    
    # Modify the final layer if num_classes is different from 1000
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

def load_checkpoint(model, checkpoint_path):
    """
    Load model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint['acc'], checkpoint['epoch'] 