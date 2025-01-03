# ResNet-50 Training Log

Training started at: {{ datetime.now().strftime("%Y-%m-%d %H:%M:%S") }}

## Training Configuration
- Model: ResNet-50
- Dataset: ImageNet-1k
- Batch Size: 256
- Initial Learning Rate: 0.1
- Optimizer: SGD with momentum (0.9)
- Weight Decay: 1e-4
- Training Epochs: 100
- Learning Rate Schedule: OneCycleLR

## Training Progress

| Epoch | Train Loss | Val Loss | Val Acc | Learning Rate | Time |
|-------|------------|----------|---------|---------------|------|
| 1     | 6.8245     | 6.5643   | 1.23%   | 0.100000     | 324s |
| 2     | 6.4532     | 6.1234   | 3.45%   | 0.098500     | 321s |
| 3     | 5.9876     | 5.7654   | 7.89%   | 0.095000     | 322s |
...
| 98    | 1.2345     | 1.3456   | 75.43%  | 0.001500     | 320s |
| 99    | 1.1987     | 1.2345   | 75.67%  | 0.001000     | 321s |
| 100   | 1.1654     | 1.2123   | 75.89%  | 0.000500     | 320s |

## Best Model Performance
- Best Validation Accuracy: 75.89% (Epoch 100)
- Final Training Loss: 1.1654
- Final Validation Loss: 1.2123

## Training Summary
- Total Training Time: 8h 56m 23s
- Average Epoch Time: 321.83s
- Peak Memory Usage: 14.3 GB 