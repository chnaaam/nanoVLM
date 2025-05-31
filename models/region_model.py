import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from models.config import VLMConfig


class YOLODetectionHead(nn.Module):
    """YOLO 스타일의 detection head"""
    def __init__(self, in_channels: int, num_classes: int, num_anchors: int):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        # 5 = x, y, w, h, objectness
        self.num_outputs = num_anchors * (5 + num_classes)
        
        self.conv = nn.Conv2d(in_channels, self.num_outputs, kernel_size=1)
        
    def forward(self, x):
        return self.conv(x)


class RegionModel(nn.Module):
    """YOLO 스타일의 Zero-shot Object Detection 모델"""
    def __init__(self, cfg: VLMConfig):
        super().__init__()
        self.cfg = cfg
        self.num_classes = cfg.detection_num_classes
        self.num_anchors = cfg.detection_num_anchors
        self.grid_sizes = cfg.detection_grid_sizes
        self.conf_threshold = cfg.detection_conf_threshold
        self.nms_threshold = cfg.detection_nms_threshold
        self.max_detections = cfg.detection_max_detections
        
        # Vision features를 detection features로 변환하는 projector
        self.feature_projector = nn.Sequential(
            nn.Linear(cfg.vit_hidden_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True)
        )
        
        # 다중 스케일 detection heads
        self.detection_heads = nn.ModuleList([
            YOLODetectionHead(512, self.num_classes, self.num_anchors)
            for _ in self.grid_sizes
        ])
        
        # Language grounding을 위한 text-visual alignment layer
        if cfg.detection_use_language_grounding:
            self.text_vision_alignment = nn.MultiheadAttention(
                embed_dim=cfg.lm_hidden_dim,
                num_heads=8,
                batch_first=True
            )
            self.class_embedding_projector = nn.Linear(cfg.lm_hidden_dim, self.num_classes)
    
    def forward(self, vision_features, text_embeddings=None):
        """
        Args:
            vision_features: [B, H*W, D] - ViT의 patch embedding들
            text_embeddings: [B, seq_len, D] - 텍스트 임베딩 (optional, zero-shot용)
        
        Returns:
            detections: List of detection outputs for each scale
        """
        batch_size = vision_features.shape[0]
        
        # Vision features를 detection features로 변환
        detection_features = self.feature_projector(vision_features)  # [B, H*W, 512]
        
        # Zero-shot을 위한 class embeddings 생성
        if text_embeddings is not None and self.cfg.detection_use_language_grounding:
            class_embeddings = self._generate_class_embeddings(text_embeddings)
        else:
            # 기본 클래스 임베딩 사용 (learnable parameters)
            class_embeddings = torch.randn(
                batch_size, self.num_classes, detection_features.shape[-1],
                device=detection_features.device
            )
        
        detections = []
        
        # 다중 스케일로 detection 수행
        for i, grid_size in enumerate(self.grid_sizes):
            # Feature map을 grid 크기로 reshape
            features_2d = self._reshape_to_grid(detection_features, grid_size)  # [B, 512, H, W]
            
            # Detection head 적용
            detection_output = self.detection_heads[i](features_2d)  # [B, num_outputs, H, W]
            
            # YOLO 형태로 reshape: [B, num_anchors, H, W, 5+num_classes]
            detection_reshaped = self._reshape_detection_output(
                detection_output, grid_size
            )
            
            detections.append(detection_reshaped)
        
        return detections, class_embeddings
    
    def _generate_class_embeddings(self, text_embeddings):
        """텍스트 임베딩으로부터 클래스 임베딩 생성"""
        # 평균 pooling으로 텍스트 임베딩을 요약
        pooled_text = torch.mean(text_embeddings, dim=1, keepdim=True)  # [B, 1, D]
        
        # 클래스 임베딩으로 투영
        class_embeddings = self.class_embedding_projector(pooled_text)  # [B, 1, num_classes]
        class_embeddings = F.softmax(class_embeddings, dim=-1)
        
        return class_embeddings
    
    def _reshape_to_grid(self, features, grid_size):
        """1D feature를 2D grid로 reshape"""
        batch_size, seq_len, feature_dim = features.shape
        
        # ViT patch 수에 맞춰 계산 (일반적으로 14x14 = 196)
        original_size = int(seq_len ** 0.5)
        
        # Grid 크기로 adaptive pooling
        features_2d = features.view(batch_size, original_size, original_size, feature_dim)
        features_2d = features_2d.permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # Target grid 크기로 resize
        features_resized = F.interpolate(
            features_2d, 
            size=(grid_size, grid_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        return features_resized
    
    def _reshape_detection_output(self, detection_output, grid_size):
        """Detection output을 YOLO 형태로 reshape"""
        batch_size = detection_output.shape[0]
        
        # [B, num_outputs, H, W] -> [B, num_anchors, H, W, 5+num_classes]
        detection_reshaped = detection_output.view(
            batch_size, self.num_anchors, 5 + self.num_classes, grid_size, grid_size
        )
        detection_reshaped = detection_reshaped.permute(0, 1, 3, 4, 2)
        
        return detection_reshaped
    
    def decode_predictions(self, detections, img_size=224):
        """Detection 결과를 bounding box로 디코딩"""
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for scale_idx, detection in enumerate(detections):
            grid_size = self.grid_sizes[scale_idx]
            batch_size, num_anchors, h, w, num_outputs = detection.shape
            
            # Grid coordinates 생성
            grid_y, grid_x = torch.meshgrid(
                torch.arange(h, device=detection.device),
                torch.arange(w, device=detection.device),
                indexing='ij'
            )
            
            for batch_idx in range(batch_size):
                batch_boxes = []
                batch_scores = []
                batch_classes = []
                
                for anchor_idx in range(num_anchors):
                    # Objectness와 class probabilities 추출
                    objectness = torch.sigmoid(detection[batch_idx, anchor_idx, :, :, 4])
                    class_probs = torch.sigmoid(detection[batch_idx, anchor_idx, :, :, 5:])
                    
                    # Confidence score = objectness * class_prob
                    max_class_probs, class_indices = torch.max(class_probs, dim=-1)
                    confidence = objectness * max_class_probs
                    
                    # Confidence threshold 적용
                    mask = confidence > self.conf_threshold
                    if not mask.any():
                        continue
                    
                    # Bounding box coordinates 추출 및 변환
                    x_center = (torch.sigmoid(detection[batch_idx, anchor_idx, :, :, 0]) + grid_x) / grid_size
                    y_center = (torch.sigmoid(detection[batch_idx, anchor_idx, :, :, 1]) + grid_y) / grid_size
                    width = torch.sigmoid(detection[batch_idx, anchor_idx, :, :, 2])
                    height = torch.sigmoid(detection[batch_idx, anchor_idx, :, :, 3])
                    
                    # 이미지 크기로 스케일링
                    x_center = x_center[mask] * img_size
                    y_center = y_center[mask] * img_size
                    width = width[mask] * img_size
                    height = height[mask] * img_size
                    
                    # x1, y1, x2, y2 형태로 변환
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    
                    boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                    scores = confidence[mask]
                    classes = class_indices[mask]
                    
                    batch_boxes.append(boxes)
                    batch_scores.append(scores)
                    batch_classes.append(classes)
                
                # Concatenate all anchors for this batch
                if batch_boxes:
                    batch_boxes = torch.cat(batch_boxes, dim=0)
                    batch_scores = torch.cat(batch_scores, dim=0)
                    batch_classes = torch.cat(batch_classes, dim=0)
                    
                    # NMS 적용
                    keep_indices = self._apply_nms(batch_boxes, batch_scores, self.nms_threshold)
                    
                    all_boxes.append(batch_boxes[keep_indices][:self.max_detections])
                    all_scores.append(batch_scores[keep_indices][:self.max_detections])
                    all_classes.append(batch_classes[keep_indices][:self.max_detections])
                else:
                    # 빈 결과
                    all_boxes.append(torch.empty((0, 4), device=detection.device))
                    all_scores.append(torch.empty((0,), device=detection.device))
                    all_classes.append(torch.empty((0,), device=detection.device, dtype=torch.long))
        
        return all_boxes, all_scores, all_classes
    
    def _apply_nms(self, boxes, scores, threshold):
        """Non-Maximum Suppression 적용"""
        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=boxes.device)
        
        # torchvision의 nms 사용
        from torchvision.ops import nms
        return nms(boxes, scores, threshold)