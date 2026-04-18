"""
Deep learning models for underwater image quality assessment.

This module provides state-of-the-art CNN and Vision Transformer models
for end-to-end quality prediction with GPU acceleration.
"""

import warnings

import timm
import torch
import torch.nn as nn
from torchvision import models


class UnderwaterQualityNet(nn.Module):
    """
    Dual-stream CNN for underwater image quality assessment.

    This network processes both RGB and LAB color spaces in parallel,
    combining features for robust quality prediction.

    Architecture:
        - ResNet50 backbone (pretrained on ImageNet)
        - Dual-stream processing (RGB + LAB)
        - Multi-task learning (quality score + blue water severity)
        - Attention mechanism for feature fusion
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_quality_levels: int = 5,
        dropout: float = 0.3
    ):
        """
        Initialize UnderwaterQualityNet.

        Parameters
        ----------
        pretrained : bool
            Use ImageNet pretrained weights
        num_quality_levels : int
            Number of quality classification levels
        dropout : float
            Dropout rate for regularization
        """
        super().__init__()

        # RGB stream - ResNet50 (fixed deprecated API)
        try:
            # PyTorch >= 0.13
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
        except ImportError:
            # PyTorch < 0.13 (legacy)
            resnet = models.resnet50(pretrained=pretrained)

        self.rgb_features = nn.Sequential(*list(resnet.children())[:-2])

        # LAB stream - ResNet50
        try:
            from torchvision.models import ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet_lab = models.resnet50(weights=weights)
        except ImportError:
            resnet_lab = models.resnet50(pretrained=pretrained)

        self.lab_features = nn.Sequential(*list(resnet_lab.children())[:-2])

        # Attention mechanism for feature fusion
        self.attention = SpatialAttention()

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Feature dimension
        feature_dim = 2048 * 2  # RGB + LAB

        # Shared feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Multi-task heads
        # Head 1: Overall quality score (0-100)
        self.quality_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Head 2: Blue water severity (0-10)
        self.blue_water_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Head 3: Feature usefulness (0-100)
        self.feature_usefulness_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        # Head 4: Quality classification
        self.quality_class_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_quality_levels)
        )

        # Head 5: Visibility score (0-1)
        self.visibility_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_lab: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass through dual-stream network.

        Parameters
        ----------
        x_rgb : torch.Tensor
            RGB input tensor (B, 3, H, W)
        x_lab : torch.Tensor
            LAB input tensor (B, 3, H, W)

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing all predictions

        Raises
        ------
        ValueError
            If input tensors have incompatible shapes
        """
        # Validate inputs
        if x_rgb.shape != x_lab.shape:
            raise ValueError(
                f"RGB and LAB tensors must have same shape. "
                f"Got RGB: {x_rgb.shape}, LAB: {x_lab.shape}"
            )

        if x_rgb.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B, C, H, W), got {x_rgb.dim()}D tensor"
            )

        batch_size = x_rgb.size(0)

        # Extract features from both streams
        rgb_feat = self.rgb_features(x_rgb)  # (B, 2048, H', W')
        lab_feat = self.lab_features(x_lab)  # (B, 2048, H', W')

        # Apply attention
        rgb_feat = self.attention(rgb_feat)
        lab_feat = self.attention(lab_feat)

        # Global pooling
        rgb_feat = self.global_pool(rgb_feat).flatten(1)  # (B, 2048)
        lab_feat = self.global_pool(lab_feat).flatten(1)  # (B, 2048)

        # Concatenate features
        combined_feat = torch.cat([rgb_feat, lab_feat], dim=1)  # (B, 4096)

        # Shared feature processing
        shared_feat = self.feature_fc(combined_feat)  # (B, 512)

        # Multi-task predictions
        quality_score = self.quality_head(shared_feat) * 100  # Scale to 0-100
        blue_water_severity = self.blue_water_head(shared_feat) * 10  # Scale to 0-10
        feature_usefulness = self.feature_usefulness_head(shared_feat) * 100
        quality_class = self.quality_class_head(shared_feat)
        visibility = self.visibility_head(shared_feat)

        # Fix squeeze issue: only squeeze last dimension if it's 1
        return {
            'quality_score': quality_score.squeeze(-1),  # Keep batch dimension
            'blue_water_severity': blue_water_severity.squeeze(-1),
            'feature_usefulness': feature_usefulness.squeeze(-1),
            'quality_class': quality_class,  # Keep 2D for classification
            'visibility': visibility.squeeze(-1)
        }


class SpatialAttention(nn.Module):
    """Spatial attention module for feature enhancement."""

    def __init__(self, kernel_size: int = 7):
        """
        Initialize spatial attention.

        Parameters
        ----------
        kernel_size : int
            Convolution kernel size (should be odd)

        Raises
        ------
        ValueError
            If kernel_size is even
        """
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.conv = nn.Conv2d(
            2, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, C, H, W)

        Returns
        -------
        torch.Tensor
            Attention-weighted features (B, C, H, W)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D tensor")

        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = self.sigmoid(self.conv(concat))  # (B, 1, H, W)

        return x * attention


class VisionTransformerQualityNet(nn.Module):
    """
    Vision Transformer-based quality assessment network.

    Uses a pretrained Vision Transformer (ViT) for feature extraction
    with specialized heads for underwater image quality prediction.
    """

    def __init__(
        self,
        model_name: str = 'vit_base_patch16_224',
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize Vision Transformer quality network.

        Parameters
        ----------
        model_name : str
            Name of ViT model from timm
        pretrained : bool
            Use pretrained weights
        dropout : float
            Dropout rate

        Raises
        ------
        RuntimeError
            If model cannot be loaded
        """
        super().__init__()

        # Load pretrained ViT
        try:
            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0  # Remove classification head
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")

        # Get feature dimension
        if not hasattr(self.vit, 'num_features'):
            raise AttributeError(
                f"Model {model_name} does not have 'num_features' attribute"
            )

        feature_dim = self.vit.num_features

        # Shared feature processing
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Multi-task heads
        self.quality_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.blue_water_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.feature_usefulness_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass through ViT network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, 3, H, W)

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing predictions
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")

        # Extract features with ViT
        features = self.vit(x)  # (B, feature_dim)

        # Process features
        shared_feat = self.feature_fc(features)  # (B, 256)

        # Multi-task predictions (fixed squeeze)
        quality_score = self.quality_head(shared_feat) * 100
        blue_water_severity = self.blue_water_head(shared_feat) * 10
        feature_usefulness = self.feature_usefulness_head(shared_feat) * 100

        return {
            'quality_score': quality_score.squeeze(-1),
            'blue_water_severity': blue_water_severity.squeeze(-1),
            'feature_usefulness': feature_usefulness.squeeze(-1)
        }


class EfficientNetQualityNet(nn.Module):
    """
    EfficientNet-based lightweight quality assessment network.

    Optimized for fast inference while maintaining high accuracy.
    """

    def __init__(
        self,
        model_name: str = 'efficientnet_b3',
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize EfficientNet quality network.

        Parameters
        ----------
        model_name : str
            EfficientNet variant
        pretrained : bool
            Use pretrained weights
        dropout : float
            Dropout rate
        """
        super().__init__()

        # Load pretrained EfficientNet
        try:
            self.efficientnet = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load {model_name}: {str(e)}")

        # Get feature dimension
        if not hasattr(self.efficientnet, 'num_features'):
            raise AttributeError(
                f"Model {model_name} does not have 'num_features' attribute"
            )

        feature_dim = self.efficientnet.num_features

        # Feature processing with squeeze-and-excitation (FIXED)
        self.se_block = SEBlock(feature_dim, reduction=16)

        # Prediction heads
        self.quality_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        self.blue_water_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.multi_metric_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10)  # Predict 10 key metrics
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass through EfficientNet."""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")

        # Extract features
        features = self.efficientnet(x)  # (B, feature_dim)

        # Apply squeeze-and-excitation
        features = self.se_block(features)

        # Predictions (fixed squeeze)
        quality_score = self.quality_head(features) * 100
        blue_water_severity = self.blue_water_head(features) * 10
        multi_metrics = self.multi_metric_head(features)

        return {
            'quality_score': quality_score.squeeze(-1),
            'blue_water_severity': blue_water_severity.squeeze(-1),
            'predicted_metrics': multi_metrics  # Keep 2D (B, 10)
        }


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    FIXED: Now properly handles 1D feature vectors from global pooling.
    """

    def __init__(self, channels: int, reduction: int = 16):
        """
        Initialize SE block.

        Parameters
        ----------
        channels : int
            Number of input channels/features
        reduction : int
            Reduction ratio

        Raises
        ------
        ValueError
            If channels < reduction
        """
        super().__init__()

        if channels < reduction:
            warnings.warn(
                f"channels ({channels}) < reduction ({reduction}), "
                f"setting reduction to {channels // 2}"
            )
            reduction = max(channels // 2, 1)

        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, C) - 1D features after global pooling

        Returns
        -------
        torch.Tensor
            Attention-weighted features (B, C)
        """
        if x.dim() != 2:
            raise ValueError(
                f"SEBlock expects 2D input (B, C), got {x.dim()}D tensor. "
                f"Apply global pooling before SE block."
            )

        # Excitation (no squeeze needed for 1D features)
        attention = self.excitation(x)  # (B, C)

        # Scale
        return x * attention


class MultiMetricPredictor(nn.Module):
    """
    Multi-metric predictor that outputs all quality metrics at once.

    This model predicts all 40+ metrics used in the traditional pipeline
    using a single forward pass, enabling ultra-fast assessment.
    """

    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        num_metrics: int = 37
    ):
        """
        Initialize multi-metric predictor.

        Parameters
        ----------
        backbone : str
            Backbone architecture ('resnet50' or 'efficientnet_b0')
        pretrained : bool
            Use pretrained weights
        num_metrics : int
            Number of metrics to predict

        Raises
        ------
        ValueError
            If backbone is not supported
        """
        super().__init__()

        # Backbone (fixed deprecated API)
        if backbone == 'resnet50':
            try:
                from torchvision.models import ResNet50_Weights
                weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
                resnet = models.resnet50(weights=weights)
            except ImportError:
                resnet = models.resnet50(pretrained=pretrained)

            self.features = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            try:
                self.features = timm.create_model(
                    'efficientnet_b0',
                    pretrained=pretrained,
                    num_classes=0
                )
                feature_dim = self.features.num_features
            except Exception as e:
                raise RuntimeError(f"Failed to load efficientnet_b0: {str(e)}")
        else:
            raise ValueError(
                f"Unsupported backbone: {backbone}. "
                f"Supported: 'resnet50', 'efficientnet_b0'"
            )

        self.backbone_name = backbone

        # Metric prediction head
        self.metric_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_metrics)
        )

        # Metric names for reference
        self.metric_names = [
            'blue_water_severity', 'turbidity_score', 'visibility_score',
            'rms_contrast', 'michelson_contrast', 'local_contrast',
            'sharpness_laplacian', 'sharpness_gradient', 'blur_estimate',
            'edge_density', 'corner_density', 'keypoint_density',
            'texture_complexity', 'uciqe_score', 'uiqm_score',
            'entropy_gray', 'entropy_color', 'hue_diversity',
            'blue_channel_mean', 'green_channel_mean', 'red_channel_mean',
            'blue_dominance', 'green_dominance', 'color_temperature',
            'weber_contrast', 'lab_contrast', 'multi_scale_edge_density',
            'blue_green_ratio', 'color_channel_std', 'histogram_uniformity_b',
            'histogram_uniformity_g', 'histogram_uniformity_r',
            'dynamic_range_b', 'dynamic_range_g', 'dynamic_range_r',
            'feature_usefulness_score', 'marine_science_value',
            'overall_score', 'blue_water_problem_severity',
            'quality_index', 'clarity_score', 'information_content'
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict all metrics.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, 3, H, W)

        Returns
        -------
        torch.Tensor
            Tensor of predicted metrics (B, num_metrics)
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D")

        features = self.features(x)  # (B, feature_dim) or (B, feature_dim, 1, 1)

        # Ensure features are 2D
        if features.dim() > 2:
            features = features.flatten(1)

        metrics = self.metric_head(features)

        # Apply sigmoid to normalize to 0-1 range
        metrics = torch.sigmoid(metrics)

        return metrics

    def predict_dict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Predict metrics and return as dictionary.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor (B, 3, H, W)

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary mapping metric names to values (B,) for each metric
        """
        metrics = self.forward(x)  # (B, num_metrics)

        return {
            name: metrics[:, i]
            for i, name in enumerate(self.metric_names)
        }
