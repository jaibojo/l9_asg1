import torch
import torch.nn as nn

class BottleneckBlock(nn.Module):
    """
    Bottleneck block used in ResNet-50.
    This block uses 1x1, 3x3, and 1x1 convolutions with a residual connection.
    Expansion factor is 4 for ResNet-50.
    
    Parameters per block:
    - 1x1 conv: (in_channels * channels) params
    - 3x3 conv: (9 * channels * channels) params
    - 1x1 conv: (channels * channels * 4) params
    - BatchNorm: (2 * channels * 2 + channels * 4) params
    - Optional downsample: (in_channels * channels * 4) params if used
    
    Feature map size: Determined by stride
    - If stride=1: Same as input size
    - If stride=2: Height and width reduced by 2
    """
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None):
        super().__init__()
        
        # First 1x1 convolution to reduce channels
        # in_channels → channels
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 3x3 convolution
        # channels → channels
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Second 1x1 convolution to increase channels
        # channels → channels * 4
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        # First bottleneck layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Second layer with 3x3 conv
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        # Third layer to expand channels
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet50(nn.Module):
    """
    Complete ResNet-50 implementation.
    
    Total Parameters: ~23.5M
    
    Input dimensions:  (batch_size, 3, 224, 224)
    Output dimensions: (batch_size, num_classes)
    
    Feature Map Dimensions (assuming 224x224 input):
    - After conv1:   (batch_size, 64, 112, 112)    # stride 2
    - After maxpool: (batch_size, 64, 56, 56)      # stride 2
    - After layer1:  (batch_size, 256, 56, 56)     # no reduction
    - After layer2:  (batch_size, 512, 28, 28)     # stride 2
    - After layer3:  (batch_size, 1024, 14, 14)    # stride 2
    - After layer4:  (batch_size, 2048, 7, 7)      # stride 2
    - After avgpool: (batch_size, 2048, 1, 1)      # global pool
    - Final output:  (batch_size, num_classes)      # fully connected
    
    Parameters per section:
    - Initial conv:   ~9K params     (7*7*3*64)
    - Layer1: ~0.2M params          (3 bottleneck blocks)
    - Layer2: ~1.2M params          (4 bottleneck blocks)
    - Layer3: ~7.1M params          (6 bottleneck blocks)
    - Layer4: ~14.9M params         (3 bottleneck blocks)
    - Final FC:  ~2M params         (2048 * num_classes)
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        # Initial layers
        self.in_channels = 64
        # Input: (3, 224, 224) -> Output: (64, 112, 112)
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        # Output: (64, 56, 56)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        # Output: (256, 56, 56)
        self.layer1 = self._make_layer(64, 3)  # 3 blocks
        # Output: (512, 28, 28)
        self.layer2 = self._make_layer(128, 4, stride=2)  # 4 blocks
        # Output: (1024, 14, 14)
        self.layer3 = self._make_layer(256, 6, stride=2)  # 6 blocks
        # Output: (2048, 7, 7)
        self.layer4 = self._make_layer(512, 3, stride=2)  # 3 blocks

        # Final layers
        # Output: (2048, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Output: (num_classes)
        self.fc = nn.Linear(512 * BottleneckBlock.expansion, num_classes)

        self._initialize_weights()

    def _make_layer(self, channels, num_blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.in_channels != channels * BottleneckBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels * BottleneckBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * BottleneckBlock.expansion)
            )

        layers = []
        # First block with potential downsampling
        layers.append(BottleneckBlock(self.in_channels, channels, stride, downsample))
        
        # Update input channels for subsequent blocks
        self.in_channels = channels * BottleneckBlock.expansion
        
        # Add remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BottleneckBlock(self.in_channels, channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """Initialize model weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Initial conv: 3x224x224 -> 64x112x112
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # Maxpool: 64x112x112 -> 64x56x56
        x = self.maxpool(x)

        # ResNet blocks
        # 64x56x56 -> 256x56x56
        x = self.layer1(x)
        # 256x56x56 -> 512x28x28
        x = self.layer2(x)
        # 512x28x28 -> 1024x14x14
        x = self.layer3(x)
        # 1024x14x14 -> 2048x7x7
        x = self.layer4(x)

        # 2048x7x7 -> 2048x1x1
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # 2048 -> num_classes
        x = self.fc(x)

        return x

def get_model(num_classes=1000, pretrained=False):
    """
    Create and return a ResNet-50 model.
    Args:
        num_classes: Number of output classes
        pretrained: Not used anymore as we're using our own implementation
    Returns:
        ResNet50 model with ~23.5M parameters
    """
    if pretrained:
        print("Warning: pretrained=True is ignored as we're using a custom ResNet-50 implementation")
    
    model = ResNet50(num_classes=num_classes)
    return model

def load_checkpoint(model, checkpoint_path):
    """
    Load model from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    return model, checkpoint['acc'], checkpoint['epoch']