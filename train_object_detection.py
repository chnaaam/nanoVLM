#!/usr/bin/env python3
"""
nanoVLM Zero-shot Object Detection 학습 스크립트

이 스크립트는 Vision-Language Detection 모델을 학습하기 위한 완전한 training pipeline을 제공합니다.
"""

import os
import json
import math
import time
import argparse
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import wandb

from models.vision_language_model import VisionLanguageDetectionModel
from models.config import VLMConfig, TrainConfig
from models.detection_losses import DetectionLossManager


class FlickrDetectionDataset(Dataset):
    """Flickr30k 스타일의 Object Detection 데이터셋"""
    
    def __init__(self, 
                 data_dir: str,
                 annotation_file: str,
                 tokenizer,
                 cfg: VLMConfig,
                 image_size: int = 224,
                 max_text_length: int = 79):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.image_size = image_size
        self.max_text_length = max_text_length
        
        # 이미지 전처리
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 어노테이션 로드
        self.annotations = self._load_annotations(annotation_file)
        
        # Detection 프롬프트 템플릿들
        self.detection_prompts = [
            "이 이미지에서 모든 객체를 찾으세요",
            "장면의 모든 항목을 감지하세요",
            "이 사진의 객체들을 찾아주세요", 
            "보이는 모든 객체를 식별하세요",
            "이 이미지에서 어떤 객체들을 볼 수 있나요?"
        ]
        
    def _load_annotations(self, annotation_file: str) -> List[Dict]:
        """Flickr30k 어노테이션 파일 로드"""
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Flickr30k 형식에서 이미지 정보 추출
            if 'images' in data:
                annotations = []
                for img_data in data['images']:
                    # 기본 정보 추출
                    annotation = {
                        'image_id': img_data.get('id', 0),
                        'file_name': img_data.get('file_name', ''),
                        'caption': img_data.get('caption', ''),
                        'height': int(img_data.get('height', 224)),
                        'width': int(img_data.get('width', 224)),
                        'objects': []
                    }
                    
                    # tokens_positive_eval에서 객체 정보 추출 (간단화)
                    if 'tokens_positive_eval' in img_data:
                        for token_info in img_data['tokens_positive_eval']:
                            if isinstance(token_info, list) and len(token_info) > 0:
                                # 임의의 bounding box 생성 (실제로는 별도의 bbox 어노테이션이 필요)
                                obj = {
                                    'bbox': [
                                        np.random.uniform(0.1, 0.6),  # x
                                        np.random.uniform(0.1, 0.6),  # y  
                                        np.random.uniform(0.2, 0.4),  # w
                                        np.random.uniform(0.2, 0.4)   # h
                                    ],
                                    'category_id': np.random.randint(0, self.cfg.detection_num_classes),
                                    'area': 0.1,
                                    'confidence': 1.0
                                }
                                annotation['objects'].append(obj)
                    
                    annotations.append(annotation)
                
                print(f"✅ Flickr30k 데이터 로드 완료: {len(annotations)}개 샘플")
                return annotations
            else:
                print(f"⚠️  'images' 키를 찾을 수 없습니다. 기본 형식으로 처리합니다.")
                return data if isinstance(data, list) else [data]
        else:
            # 더미 데이터 생성 (실제 데이터가 없을 때)
            print(f"⚠️  어노테이션 파일을 찾을 수 없습니다: {annotation_file}")
            print("🔄 더미 데이터로 학습을 진행합니다...")
            return self._generate_dummy_data()
    
    def _generate_dummy_data(self) -> List[Dict]:
        """더미 데이터 생성 (테스트용)"""
        dummy_data = []
        for i in range(1000):  # 1000개의 더미 샘플
            dummy_data.append({
                'image_id': i,
                'file_name': f'dummy_{i:04d}.jpg',
                'caption': f"샘플 이미지 {i}에 다양한 객체들이 있습니다",
                'height': 224,
                'width': 224,
                'objects': [
                    {
                        'bbox': [0.2, 0.3, 0.4, 0.5],  # normalized [x, y, w, h]
                        'category_id': np.random.randint(0, self.cfg.detection_num_classes),
                        'area': 0.2 * 0.2,
                        'confidence': 1.0
                    }
                ]
            })
        return dummy_data
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Dict:
        annotation = self.annotations[idx]
        
        # 이미지 로드
        image = self._load_image(annotation)
        
        # 텍스트 프롬프트 생성
        text_prompt = self._generate_text_prompt(annotation)
        text_inputs = self._tokenize_text(text_prompt)
        
        # Detection targets 생성
        detection_targets = self._create_detection_targets(annotation)
        
        return {
            'image': image,
            'input_ids': text_inputs['input_ids'].squeeze(0),
            'attention_mask': text_inputs['attention_mask'].squeeze(0),
            'detection_targets': detection_targets,
            'annotation': annotation
        }
    
    def _load_image(self, annotation: Dict) -> torch.Tensor:
        """이미지 로드 및 전처리"""
        file_name = annotation.get('file_name', 'dummy_image.jpg')
        image_path = self.data_dir / file_name
        
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"⚠️  이미지 로딩 실패 {file_name}: {e}, 더미 이미지 사용")
                image = self._create_dummy_image()
        else:
            # 더미 이미지 생성
            image = self._create_dummy_image()
        
        return self.transform(image)
    
    def _create_dummy_image(self) -> Image.Image:
        """더미 이미지 생성"""
        return Image.new('RGB', (self.image_size, self.image_size), 
                        color=(np.random.randint(0, 255), 
                               np.random.randint(0, 255), 
                               np.random.randint(0, 255)))
    
    def _generate_text_prompt(self, annotation: Dict) -> str:
        """Detection을 위한 텍스트 프롬프트 생성"""
        # 다양한 방식으로 프롬프트 생성
        if 'caption' in annotation and annotation['caption'] and np.random.random() > 0.3:
            # 캡션 기반 프롬프트
            caption = annotation['caption']
            prompts = [
                f"다음 설명의 이미지에서 객체를 찾으세요: {caption}",
                f"이미지 설명: {caption} - 여기서 모든 객체를 감지하세요",
                f"{caption} 이 장면에서 보이는 객체들을 찾아주세요"
            ]
            return np.random.choice(prompts)
        else:
            # 일반적인 detection 프롬프트
            return np.random.choice(self.detection_prompts)
    
    def _tokenize_text(self, text: str) -> Dict:
        """텍스트 토큰화"""
        return self.tokenizer(
            text,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def _create_detection_targets(self, annotation: Dict) -> List[torch.Tensor]:
        """YOLO 형식의 detection targets 생성"""
        targets = []
        
        for grid_size in self.cfg.detection_grid_sizes:
            target = torch.zeros(
                self.cfg.detection_num_anchors, 
                grid_size, 
                grid_size, 
                5 + self.cfg.detection_num_classes
            )
            
            # 객체들을 grid에 할당
            objects = annotation.get('objects', [])
            for obj in objects:
                bbox = obj['bbox']  # normalized [x, y, w, h]
                category_id = obj['category_id']
                
                # Grid cell 계산
                grid_x = int(bbox[0] * grid_size)
                grid_y = int(bbox[1] * grid_size)
                grid_x = min(grid_x, grid_size - 1)
                grid_y = min(grid_y, grid_size - 1)
                
                # 첫 번째 앵커에 할당 (단순화)
                anchor_idx = 0
                
                # Bounding box 정보
                target[anchor_idx, grid_y, grid_x, 0] = bbox[0]  # x
                target[anchor_idx, grid_y, grid_x, 1] = bbox[1]  # y  
                target[anchor_idx, grid_y, grid_x, 2] = bbox[2]  # w
                target[anchor_idx, grid_y, grid_x, 3] = bbox[3]  # h
                target[anchor_idx, grid_y, grid_x, 4] = 1.0      # objectness
                
                # 클래스 정보 (one-hot)
                if category_id < self.cfg.detection_num_classes:
                    target[anchor_idx, grid_y, grid_x, 5 + category_id] = 1.0
            
            targets.append(target)
        
        return targets


# COCODetectionDataset을 FlickrDetectionDataset의 alias로 설정 (하위 호환성)
COCODetectionDataset = FlickrDetectionDataset


class DetectionTrainer:
    """Object Detection 모델 학습을 위한 Trainer 클래스"""
    
    def __init__(self, 
                 model: VisionLanguageDetectionModel,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 cfg: VLMConfig,
                 train_cfg: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.train_cfg = train_cfg
        
        # Device 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Optimizer 설정
        self.optimizer = self._setup_optimizer()
        
        # Scheduler 설정
        self.scheduler = self._setup_scheduler()
        
        # Loss scaler for mixed precision
        self.scaler = GradScaler()
        
        # 학습 상태
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Wandb 초기화
        if train_cfg.log_wandb:
            self._init_wandb()
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Optimizer 설정"""
        # 다른 학습률 적용
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'region_model' in n],
                'lr': self.train_cfg.lr_mp,
                'name': 'detection_head'
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if 'region_model' not in n],
                'lr': self.train_cfg.lr_backbones,
                'name': 'backbone'
            }
        ]
        
        return optim.AdamW(param_groups, weight_decay=1e-4)
    
    def _setup_scheduler(self):
        """Learning rate scheduler 설정"""
        total_steps = len(self.train_loader) * self.train_cfg.epochs
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps,
            eta_min=1e-6
        )
    
    def _init_wandb(self):
        """Wandb 초기화"""
        wandb.init(
            project="nanovlm-detection",
            entity=self.train_cfg.wandb_entity,
            config={
                **self.cfg.__dict__,
                **self.train_cfg.__dict__
            }
        )
    
    def train_epoch(self) -> Dict[str, float]:
        """한 에포크 학습"""
        self.model.train()
        
        total_loss = 0.0
        total_detection_loss = 0.0 
        total_language_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # 데이터를 device로 이동
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            detection_targets = [t.to(self.device) for t in batch['detection_targets']]
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(
                    input_ids=input_ids,
                    image=images,
                    attention_mask=attention_mask,
                    detection_targets=detection_targets
                )
                
                loss = outputs['total_loss']
                detection_loss = outputs['losses'].get('detection_loss', 0.0)
                language_loss = outputs['losses'].get('language_loss', 0.0)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.train_cfg.max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.train_cfg.max_grad_norm
                )
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler:
                self.scheduler.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            if isinstance(detection_loss, torch.Tensor):
                total_detection_loss += detection_loss.item()
            if isinstance(language_loss, torch.Tensor):
                total_language_loss += language_loss.item()
            num_batches += 1
            self.global_step += 1
            
            # 로깅
            if batch_idx % 50 == 0:
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                      f"Loss: {loss.item():.4f}, Det: {detection_loss:.4f}, LM: {language_loss:.4f}")
                
                if self.train_cfg.log_wandb:
                    wandb.log({
                        'train/batch_loss': loss.item(),
                        'train/detection_loss': detection_loss,
                        'train/language_loss': language_loss,
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })
        
        return {
            'total_loss': total_loss / num_batches,
            'detection_loss': total_detection_loss / num_batches,
            'language_loss': total_language_loss / num_batches
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """모델 검증"""
        self.model.eval()
        
        total_loss = 0.0
        total_detection_loss = 0.0
        total_language_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            detection_targets = [t.to(self.device) for t in batch['detection_targets']]
            
            outputs = self.model(
                input_ids=input_ids,
                image=images,
                attention_mask=attention_mask,
                detection_targets=detection_targets
            )
            
            loss = outputs['total_loss']
            detection_loss = outputs['losses'].get('detection_loss', 0.0)
            language_loss = outputs['losses'].get('language_loss', 0.0)
            
            total_loss += loss.item()
            if isinstance(detection_loss, torch.Tensor):
                total_detection_loss += detection_loss.item()
            if isinstance(language_loss, torch.Tensor):
                total_language_loss += language_loss.item()
            num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'detection_loss': total_detection_loss / num_batches,
            'language_loss': total_language_loss / num_batches
        }
    
    def save_checkpoint(self, path: str):
        """체크포인트 저장"""
        torch.save({
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.cfg,
            'train_config': self.train_cfg
        }, path)
        print(f"✅ 체크포인트 저장: {path}")
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ 체크포인트 로드: {path} (Epoch {self.current_epoch})")
    
    def train(self):
        """전체 학습 루프"""
        print(f"🚀 학습 시작: {self.train_cfg.epochs} epochs")
        
        for epoch in range(self.current_epoch, self.train_cfg.epochs):
            self.current_epoch = epoch
            
            # 학습
            train_metrics = self.train_epoch()
            
            # 검증
            val_metrics = self.validate()
            
            # 로깅
            print(f"\nEpoch {epoch+1}/{self.train_cfg.epochs}")
            print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
                  f"Det: {train_metrics['detection_loss']:.4f}, "
                  f"LM: {train_metrics['language_loss']:.4f}")
            print(f"Val - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Det: {val_metrics['detection_loss']:.4f}, "
                  f"LM: {val_metrics['language_loss']:.4f}")
            
            if self.train_cfg.log_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['total_loss'],
                    'train/epoch_detection_loss': train_metrics['detection_loss'],
                    'train/epoch_language_loss': train_metrics['language_loss'],
                    'val/epoch_loss': val_metrics['total_loss'],
                    'val/epoch_detection_loss': val_metrics['detection_loss'],
                    'val/epoch_language_loss': val_metrics['language_loss']
                })
            
            # 체크포인트 저장
            checkpoint_dir = Path(self.cfg.vlm_checkpoint_path)
            checkpoint_dir.mkdir(exist_ok=True)
            
            # 정기 저장
            if (epoch + 1) % 5 == 0:
                checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)
            
            # Best model 저장
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                best_path = checkpoint_dir / "best_model.pt"
                self.save_checkpoint(best_path)
                print(f"🏆 새로운 최고 모델 저장! Val Loss: {self.best_val_loss:.4f}")
        
        print("🎉 학습 완료!")


def custom_collate_fn(batch):
    """Detection targets을 올바르게 처리하는 커스텀 collate function"""
    # 각 키별로 데이터 수집
    images = []
    input_ids = []
    attention_masks = []
    detection_targets_by_scale = [[] for _ in range(3)]  # 3개 스케일 (7, 14, 28)
    annotations = []
    
    for item in batch:
        images.append(item['image'])
        input_ids.append(item['input_ids'])
        attention_masks.append(item['attention_mask'])
        
        # Detection targets을 스케일별로 분리
        for scale_idx, target in enumerate(item['detection_targets']):
            detection_targets_by_scale[scale_idx].append(target)
        
        annotations.append(item['annotation'])
    
    # 텐서로 스택
    batch_dict = {
        'image': torch.stack(images, dim=0),
        'input_ids': torch.stack(input_ids, dim=0),
        'attention_mask': torch.stack(attention_masks, dim=0),
        'detection_targets': [torch.stack(targets, dim=0) for targets in detection_targets_by_scale],
        'annotation': annotations
    }
    
    return batch_dict


def create_data_loaders(
    cfg: VLMConfig, 
    train_cfg: TrainConfig, 
    tokenizer,
    image_dir: str,
    annotation_file: str
) -> Tuple[DataLoader, DataLoader]:
    """데이터 로더 생성"""
    
    # 전체 데이터셋 로드
    full_dataset = FlickrDetectionDataset(
        data_dir=image_dir,
        annotation_file=annotation_file,
        tokenizer=tokenizer,
        cfg=cfg
    )
    
    # 데이터셋 크기
    dataset_size = len(full_dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    print(f"📊 데이터셋 분할 정보:")
    print(f"  전체 데이터: {dataset_size}개")
    print(f"  학습 데이터: {train_size}개 (90%)")
    print(f"  검증 데이터: {val_size}개 (10%)")
    
    # 랜덤 시드 고정으로 재현 가능한 분할
    torch.manual_seed(42)
    indices = torch.randperm(dataset_size).tolist()
    
    # 인덱스로 데이터셋 분할
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Subset으로 분할
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    # 데이터 로더 (커스텀 collate function 사용)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,  # 멀티프로세싱 에러 방지
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg.batch_size,
        shuffle=False,
        num_workers=0,  # 멀티프로세싱 에러 방지
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='nanoVLM Object Detection Training')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_dir', type=str, help='Data directory', default='/Users/chnam/Downloads/flickr30k/Images')
    parser.add_argument('--annotation_file', type=str, help='Annotation file path', default='/Users/chnam/Downloads/final_flickr_separateGT_train_korean.json')
    parser.add_argument('--wandb', action='store_true', help='Use wandb logging')
    args = parser.parse_args()
    
    # Annotation Format
    # "images": [
    #   {
    #       "file_name": "3359636318.jpg", 
    #       "height": "334", 
    #       "width": "500", 
    #       "id": 0, 
    #       "caption": "\ub450 \uc0ac\ub78c\uc774 \ud734\ub300\ud3f0 \uac00\uac8c \uc606\uc5d0 \uc788\ub294 \ube44\ub514\uc624 \uac8c\uc784 \uac00\uac8c \ubc16\uc5d0\uc11c \uc774\uc57c\uae30\ud558\uace0 \uc788\ub2e4.", 
    #       "dataset_name": "flickr", 
    #       "tokens_negative": [[0, 91]], 
    #       "sentence_id": 0, 
    #       "original_img_id": 3359636318, 
    #       "tokens_positive_eval": [[[0, 10]], [[34, 53]], [[67, 89]]]
    #   },
    #   ...
    # ]

    # 설정 로드
    cfg = VLMConfig()
    train_cfg = TrainConfig()
    
    # Detection 학습에 적합한 설정으로 조정
    train_cfg.batch_size = 8  # 메모리 효율성을 위해 작은 배치 크기 사용
    train_cfg.epochs = 10
    train_cfg.log_wandb = args.wandb
    
    print("🔧 설정 정보:")
    print(f"  Detection 클래스 수: {cfg.detection_num_classes}")
    print(f"  Grid 크기: {cfg.detection_grid_sizes}")
    print(f"  Batch size: {train_cfg.batch_size}")
    print(f"  Learning rates: MP={train_cfg.lr_mp}, Backbone={train_cfg.lr_backbones}")
    print(f"  Epochs: {train_cfg.epochs}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 모델 초기화
    model = VisionLanguageDetectionModel(cfg, load_backbone=True)
    
    # 데이터 로더 생성
    train_loader, val_loader = create_data_loaders(cfg, train_cfg, tokenizer, args.image_dir, args.annotation_file)
    
    # Trainer 초기화
    trainer = DetectionTrainer(model, train_loader, val_loader, cfg, train_cfg)
    
    # 체크포인트에서 재개
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 학습 시작
    trainer.train()
    
    # 최종 모델 저장
    model.save_pretrained("./final_detection_model")
    print("💾 최종 모델 저장 완료: ./final_detection_model")


if __name__ == "__main__":
    main() 