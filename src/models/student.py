"""
Lightweight student model for deepfake detection
"""
import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()
        
        # Lightweight backbone - MobileNetV3 small
        # For now, we'll use a simple CNN structure
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            # Depthwise separable conv blocks
            nn.Conv2d(16, 16, 3, stride=1, padding=1, groups=16),
            nn.Conv2d(16, 32, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32),
            nn.Conv2d(32, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, 3, stride=2, padding=1, groups=64),
            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Test model
    model = StudentModel()
    test_input = torch.randn(1, 3, 224, 224)
    output = model(test_input)
    print(f"Model output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")