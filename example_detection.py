#!/usr/bin/env python3
"""
nanoVLM Zero-shot Object Detection 예제

이 스크립트는 새로 구현된 YOLO 스타일의 Zero-shot Object Detection 모델의 사용법을 보여줍니다.
"""

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoTokenizer

from models.vision_language_model import VisionLanguageDetectionModel
from models.config import VLMConfig


def load_and_prepare_image(image_path, image_size=224):
    """이미지를 로드하고 모델 입력용으로 전처리합니다."""
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 평균
            std=[0.229, 0.224, 0.225]   # ImageNet 표준편차
        )
    ])
    
    return transform(image).unsqueeze(0)  # 배치 차원 추가


def prepare_text_input(text, tokenizer, max_length=79):
    """텍스트를 토큰화하고 모델 입력용으로 준비합니다."""
    encoded = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoded['input_ids'], encoded['attention_mask']


def main():
    # 1. 설정 및 모델 로드
    print("🔄 모델 설정 중...")
    cfg = VLMConfig()
    
    # Detection 설정 조정
    cfg.detection_conf_threshold = 0.3
    cfg.detection_nms_threshold = 0.5
    cfg.detection_max_detections = 50
    
    # 모델 초기화 (이 예제에서는 사전 훈련된 가중치 없이)
    model = VisionLanguageDetectionModel(cfg, load_backbone=False)
    model.eval()
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_tokenizer)
    
    # 패딩 토큰 설정 (EOS 토큰을 패딩 토큰으로 사용)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"✅ 패딩 토큰을 EOS 토큰으로 설정: {tokenizer.pad_token}")
    
    print("✅ 모델 로드 완료!")

    # 2. 예제 이미지 준비 (실제 사용시에는 실제 이미지 경로를 사용)
    print("\n🖼️  예제 이미지 생성 중...")
    # 더미 이미지 (실제로는 실제 이미지를 사용해야 함)
    dummy_image = torch.randn(1, 3, 224, 224)
    
    # 3. Zero-shot Object Detection 예제들
    detection_prompts = [
        "Find all cats and dogs in this image",
        "Detect cars and motorcycles",
        "Locate all people in the scene",
        "Find furniture like chairs and tables"
    ]
    
    print("\n🎯 Zero-shot Object Detection 실행 중...")
    
    for i, prompt in enumerate(detection_prompts, 1):
        print(f"\n--- 예제 {i}: {prompt} ---")
        
        # 텍스트 입력 준비
        input_ids, attention_mask = prepare_text_input(prompt, tokenizer)
        
        try:
            # Object Detection 수행
            with torch.no_grad():
                boxes, scores, classes = model.detect_objects(
                    input_ids=input_ids,
                    image=dummy_image,
                    attention_mask=attention_mask,
                    conf_threshold=0.3,
                    nms_threshold=0.5
                )
            
            # 결과 출력
            if len(boxes) > 0 and len(boxes[0]) > 0:
                print(f"✅ {len(boxes[0])}개 객체 탐지됨!")
                for j, (box, score, cls) in enumerate(zip(boxes[0][:5], scores[0][:5], classes[0][:5])):
                    print(f"  객체 {j+1}: 클래스={cls.item()}, 신뢰도={score:.3f}, "
                          f"박스=[{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
            else:
                print("❌ 탐지된 객체 없음")
                
        except Exception as e:
            print(f"❌ 오류 발생: {e}")

    # 4. 텍스트 생성과 동시에 Object Detection 수행
    print(f"\n📝 텍스트 생성 + Object Detection 예제...")
    
    generation_prompt = "Describe what you see in this image:"
    input_ids, attention_mask = prepare_text_input(generation_prompt, tokenizer)
    
    try:
        with torch.no_grad():
            generated_tokens, detection_results = model.generate_with_detection(
                input_ids=input_ids,
                image=dummy_image,
                attention_mask=attention_mask,
                max_new_tokens=30,
                return_detections=True
            )
        
        # 생성된 텍스트 디코딩
        generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        print(f"생성된 텍스트: {generated_text}")
        
        # Detection 결과
        if detection_results and len(detection_results['boxes']) > 0:
            num_detected = len(detection_results['boxes'][0])
            print(f"동시에 {num_detected}개 객체 탐지됨!")
        else:
            print("텍스트 생성 중 객체 탐지되지 않음")
            
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

    # 5. 모델 구조 정보
    print(f"\n📊 모델 정보:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  총 파라미터 수: {total_params:,}")
    print(f"  훈련 가능한 파라미터 수: {trainable_params:,}")
    print(f"  Detection 클래스 수: {cfg.detection_num_classes}")
    print(f"  Grid 크기들: {cfg.detection_grid_sizes}")
    
    print(f"\n🎉 모든 예제 완료!")


if __name__ == "__main__":
    main() 