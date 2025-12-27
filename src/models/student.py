import torch
import torch.nn as nn
import torchvision.models as models

class StudentModel(nn.Module):
    """Lightweight student model for deepfake detection"""
    
    def __init__(self, num_classes=2):
        super(StudentModel, self).__init__()
        
        # Use MobileNetV3 as backbone for efficiency
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        
        # Replace classifier with our own
        in_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

class MultiModalStudent(nn.Module):
    """Multimodal student for video + audio"""
    
    def __init__(self, num_classes=2):
        super(MultiModalStudent, self).__init__()
        
        # Video branch
        self.video_backbone = models.mobilenet_v3_small(pretrained=True)
        video_features = self.video_backbone.classifier[0].in_features
        self.video_backbone.classifier = nn.Identity()
        
        # Audio branch (simple CNN for spectrograms)
        self.audio_branch = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(video_features + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, video, audio=None):
        video_feat = self.video_backbone(video)
        
        if audio is not None:
            audio_feat = self.audio_branch(audio)
            combined = torch.cat([video_feat, audio_feat], dim=1)
            return self.fusion(combined)
        else:
            # Video only mode
            return self.fusion(torch.cat([video_feat, torch.zeros(video_feat.size(0), 64).to(video_feat.device)], dim=1))

if __name__ == "__main__":
    # Test the models
    model = StudentModel()
    print(f"Student model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    multimodal = MultiModalStudent()
    print(f"Multimodal model parameters: {sum(p.numel() for p in multimodal.parameters()):,}")
    
    # Test forward pass
    dummy_video = torch.randn(1, 3, 224, 224)
    dummy_audio = torch.randn(1, 1, 128, 128)
    
    with torch.no_grad():
        output1 = model(dummy_video)
        output2 = multimodal(dummy_video, dummy_audio)
        print(f"Video-only output shape: {output1.shape}")
        print(f"Multimodal output shape: {output2.shape}")