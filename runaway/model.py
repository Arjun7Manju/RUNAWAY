import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    Standard Convolutional Block:
    Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ObjectDetectorCNN(nn.Module):
    """
    Convolutional Neural Network for Object Detection.
    
    Architecture Design:
    1. CNN Backbone (Feature Extractor):
       Hierarchically extracts edges, textures, patterns, and shape features.
    2. Shared Dense Layer:
       Combines spatial feature maps into a semantic representation.
    3. Dual Task Heads:
       - Classification Head: Predicts object class probabilities (C classes).
       - Bounding Box Regression Head: Predicts normalized [x1, y1, x2, y2] coordinates.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes

        # -------------------------------------------------------------
        # 1. Feature Extractor Backbone (CNN)
        # -------------------------------------------------------------
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 16),   # (B, 16, H/2, W/2)
            ConvBlock(16, 32),            # (B, 32, H/4, W/4)
            ConvBlock(32, 64),            # (B, 64, H/8, W/8)
            ConvBlock(64, 128),           # (B, 128, H/16, W/16)
        )

        # Spatial pooling produces fixed 2x2 spatial map regardless of input size
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        flatten_dim = 128 * 2 * 2  # 512

        # -------------------------------------------------------------
        # 2. Shared Semantic Feature Layer
        # -------------------------------------------------------------
        self.shared_fc = nn.Sequential(
            nn.Linear(flatten_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        # -------------------------------------------------------------
        # 3. Dual Detection Heads
        # -------------------------------------------------------------
        # Head A: Object Classification (Outputs class logits)
        self.classifier_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

        # Head B: Bounding Box Regression (Outputs [x1, y1, x2, y2] in [0, 1])
        self.box_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),
            nn.Sigmoid()  # Bound bounding box predictions to [0, 1]
        )

    def forward(self, x: torch.Tensor):
        """
        Forward pass.
        Args:
            x: Tensor of shape (Batch_Size, Channels, Height, Width)
        Returns:
            class_logits: Tensor of shape (Batch_Size, num_classes)
            box_coords: Tensor of shape (Batch_Size, 4) in [x1, y1, x2, y2] format
        """
        features = self.backbone(x)
        pooled = self.pool(features)
        flat = torch.flatten(pooled, start_dim=1)
        shared = self.shared_fc(flat)

        class_logits = self.classifier_head(shared)
        box_coords = self.box_head(shared)

        return class_logits, box_coords
