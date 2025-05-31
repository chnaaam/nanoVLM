import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class YOLOLoss(nn.Module):
    """YOLO 스타일의 detection loss"""
    
    def __init__(self, 
                 num_classes: int,
                 num_anchors: int,
                 lambda_coord: float = 5.0,
                 lambda_noobj: float = 0.5,
                 lambda_obj: float = 1.0,
                 lambda_class: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.lambda_coord = lambda_coord  # Coordinate loss weight
        self.lambda_noobj = lambda_noobj  # No object loss weight  
        self.lambda_obj = lambda_obj      # Object loss weight
        self.lambda_class = lambda_class  # Classification loss weight
        
    def forward(self, predictions: List[torch.Tensor], targets: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            predictions: List of detection outputs [B, num_anchors, H, W, 5+num_classes] for each scale
            targets: List of target tensors [B, num_anchors, H, W, 5+num_classes] for each scale
        
        Returns:
            total_loss: Combined detection loss
        """
        total_loss = 0.0
        
        for pred, target in zip(predictions, targets):
            # pred, target shape: [B, num_anchors, H, W, 5+num_classes]
            batch_size, num_anchors, h, w, num_outputs = pred.shape
            
            # Extract components
            pred_coords = pred[..., :4]      # [B, A, H, W, 4] - x, y, w, h
            pred_conf = pred[..., 4]         # [B, A, H, W] - objectness
            pred_classes = pred[..., 5:]     # [B, A, H, W, num_classes]
            
            target_coords = target[..., :4]
            target_conf = target[..., 4]
            target_classes = target[..., 5:]
            
            # Object mask: 1 if object exists, 0 otherwise
            obj_mask = target_conf > 0  # [B, A, H, W]
            noobj_mask = ~obj_mask
            
            # 1. Coordinate Loss (only for cells with objects)
            coord_loss = 0.0
            if obj_mask.any():
                # Apply sigmoid to predictions
                pred_xy = torch.sigmoid(pred_coords[..., :2])  # x, y
                pred_wh = torch.sigmoid(pred_coords[..., 2:])  # w, h
                
                target_xy = target_coords[..., :2]
                target_wh = target_coords[..., 2:]
                
                # Coordinate loss (MSE)
                xy_loss = F.mse_loss(
                    pred_xy[obj_mask], 
                    target_xy[obj_mask], 
                    reduction='sum'
                )
                wh_loss = F.mse_loss(
                    pred_wh[obj_mask], 
                    target_wh[obj_mask], 
                    reduction='sum'
                )
                
                coord_loss = xy_loss + wh_loss
            
            # 2. Objectness Loss
            pred_conf_sigmoid = torch.sigmoid(pred_conf)
            
            # Object confidence loss
            obj_conf_loss = F.binary_cross_entropy(
                pred_conf_sigmoid[obj_mask],
                target_conf[obj_mask],
                reduction='sum'
            ) if obj_mask.any() else 0.0
            
            # No-object confidence loss
            noobj_conf_loss = F.binary_cross_entropy(
                pred_conf_sigmoid[noobj_mask],
                target_conf[noobj_mask],
                reduction='sum'
            ) if noobj_mask.any() else 0.0
            
            # 3. Classification Loss (only for cells with objects)
            class_loss = 0.0
            if obj_mask.any():
                pred_classes_sigmoid = torch.sigmoid(pred_classes)
                class_loss = F.binary_cross_entropy(
                    pred_classes_sigmoid[obj_mask],
                    target_classes[obj_mask],
                    reduction='sum'
                )
            
            # Total number of objects for normalization
            num_objects = obj_mask.sum().float()
            if num_objects > 0:
                coord_loss /= num_objects
                obj_conf_loss /= num_objects
                class_loss /= num_objects
            
            # Normalize no-object loss by total predictions
            total_predictions = batch_size * num_anchors * h * w
            noobj_conf_loss /= total_predictions
            
            # Combine losses
            scale_loss = (
                self.lambda_coord * coord_loss +
                self.lambda_obj * obj_conf_loss +
                self.lambda_noobj * noobj_conf_loss +
                self.lambda_class * class_loss
            )
            
            total_loss += scale_loss
        
        return total_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Predictions [N, C] 
            targets: Ground truth [N, C] (one-hot encoded)
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)
        
        # Focal loss formula
        focal_weight = self.alpha * (1 - p) ** self.gamma
        loss = focal_weight * F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class DIoULoss(nn.Module):
    """Distance-IoU Loss for better bounding box regression"""
    
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps
        
    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_boxes: Predicted boxes [N, 4] in format [x1, y1, x2, y2]
            target_boxes: Target boxes [N, 4] in format [x1, y1, x2, y2]
        """
        # Calculate intersection
        inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        target_area = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
        union_area = pred_area + target_area - inter_area
        
        # IoU
        iou = inter_area / (union_area + self.eps)
        
        # Calculate center distance
        pred_center_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        pred_center_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        target_center_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
        target_center_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
        
        center_distance = (pred_center_x - target_center_x) ** 2 + (pred_center_y - target_center_y) ** 2
        
        # Calculate diagonal of smallest enclosing box
        enclose_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
        enclose_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
        enclose_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
        enclose_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
        
        diagonal_distance = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
        
        # DIoU Loss
        diou = iou - center_distance / (diagonal_distance + self.eps)
        loss = 1 - diou
        
        return loss.mean()


class DetectionLossManager:
    """다양한 detection loss를 관리하는 클래스"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.yolo_loss = YOLOLoss(
            num_classes=cfg.detection_num_classes,
            num_anchors=cfg.detection_num_anchors
        )
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        self.diou_loss = DIoULoss()
        
    def compute_total_loss(self, predictions, targets, loss_type='yolo'):
        """
        여러 loss 타입 중 선택하여 계산
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets  
            loss_type: 'yolo', 'focal', 'diou', or 'combined'
        """
        if loss_type == 'yolo':
            return self.yolo_loss(predictions, targets)
        elif loss_type == 'focal':
            # Focal loss 적용 (클래스 불균형 해결용)
            total_loss = 0.0
            for pred, target in zip(predictions, targets):
                # Extract class predictions and targets
                pred_classes = pred[..., 5:]
                target_classes = target[..., 5:]
                obj_mask = target[..., 4] > 0
                
                if obj_mask.any():
                    loss = self.focal_loss(
                        pred_classes[obj_mask].view(-1, self.cfg.detection_num_classes),
                        target_classes[obj_mask].view(-1, self.cfg.detection_num_classes)
                    )
                    total_loss += loss
            return total_loss
        elif loss_type == 'combined':
            # 여러 loss 조합
            yolo_loss = self.yolo_loss(predictions, targets)
            focal_loss = self.compute_total_loss(predictions, targets, 'focal')
            return yolo_loss + 0.1 * focal_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}") 