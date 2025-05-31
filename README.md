# nanoVLM - Zero-shot Object Detection 🎯

**nanoVLM**는 최소한의 코드로 구현된 경량 Zero-shot Object Detection 모델입니다. Vision-Language 이해와 YOLO 스타일의 객체 탐지를 결합하여, 자연어 설명을 통한 zero-shot 객체 탐지를 수행할 수 있습니다.

## 🌟 주요 특징

- **Zero-shot Object Detection**: 자연어 프롬프트를 통한 객체 탐지
- **YOLO 스타일 아키텍처**: 다중 스케일 detection heads
- **Vision-Language 통합**: ViT encoder + Language Model + Detection Head
- **경량 설계**: 순수 PyTorch로 구현된 간단한 아키텍처
- **유연한 Loss 함수**: YOLO Loss, Focal Loss, DIoU Loss 지원

## 🏗️ 모델 구조

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   ViT Encoder   │───►│ Modality Projector│───►│ Language Model  │
│  (SigLIP-Base)  │    │                  │    │  (SmolLM2-135M) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                                               │
         ▼                                               ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Region Model    │◄───│ Text Embeddings  │    │ Text Generation │
│ (YOLO-style)    │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Detections    │
│ (Boxes, Classes)│
└─────────────────┘
```

## 🚀 빠른 시작

### 설치

```bash
pip install torch torchvision transformers safetensors huggingface_hub pillow
```

### 기본 사용법

```python
from models.vision_language_model import VisionLanguageDetectionModel
from models.config import VLMConfig
import torch
from PIL import Image
from transformers import AutoTokenizer

# 모델 설정
cfg = VLMConfig()
model = VisionLanguageDetectionModel(cfg, load_backbone=False)
tokenizer = AutoTokenizer.from_pretrained(cfg.lm_tokenizer)

# 이미지 로드
image = torch.randn(1, 3, 224, 224)  # 실제로는 실제 이미지 사용

# Zero-shot Object Detection
prompt = "Find all cats and dogs in this image"
input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']

with torch.no_grad():
    boxes, scores, classes = model.detect_objects(input_ids, image)

print(f"탐지된 객체 수: {len(boxes[0])}")
```

### 텍스트 생성 + 객체 탐지

```python
# 텍스트 생성과 동시에 객체 탐지
prompt = "Describe what you see:"
input_ids = tokenizer(prompt, return_tensors='pt')['input_ids']

with torch.no_grad():
    generated_tokens, detections = model.generate_with_detection(
        input_ids, image, max_new_tokens=50, return_detections=True
    )

# 결과 확인
generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"생성된 텍스트: {generated_text}")
print(f"탐지된 객체 수: {len(detections['boxes'][0])}")
```

## 📊 모델 설정

주요 설정 옵션들:

```python
from models.config import VLMConfig

cfg = VLMConfig()

# Detection 관련 설정
cfg.detection_num_classes = 80        # COCO 클래스 수
cfg.detection_num_anchors = 3         # 앵커 박스 수
cfg.detection_grid_sizes = (7, 14, 28)  # 다중 스케일 grid
cfg.detection_conf_threshold = 0.5    # Confidence threshold
cfg.detection_nms_threshold = 0.4     # NMS threshold
cfg.detection_use_language_grounding = True  # 언어 기반 grounding
```

## 🎯 Loss 함수

다양한 detection loss 함수를 지원합니다:

```python
from models.detection_losses import DetectionLossManager

loss_manager = DetectionLossManager(cfg)

# YOLO Loss
yolo_loss = loss_manager.compute_total_loss(predictions, targets, 'yolo')

# Focal Loss (클래스 불균형 해결)
focal_loss = loss_manager.compute_total_loss(predictions, targets, 'focal')

# 조합된 Loss
combined_loss = loss_manager.compute_total_loss(predictions, targets, 'combined')
```

## 📝 예제 실행

완전한 예제를 실행해보세요:

```bash
python example_detection.py
```

이 스크립트는 다음을 포함합니다:

- Zero-shot object detection 예제들
- 텍스트 생성과 동시 객체 탐지
- 모델 정보 출력

## 🔧 커스터마이징

### 새로운 Detection Head 추가

```python
from models.region_model import YOLODetectionHead

# 커스텀 detection head
class CustomDetectionHead(nn.Module):
    def __init__(self, in_channels, num_classes, num_anchors):
        super().__init__()
        # 여기에 커스텀 구현
        pass
```

### 손실 함수 커스터마이징

```python
from models.detection_losses import YOLOLoss

class CustomLoss(YOLOLoss):
    def forward(self, predictions, targets):
        # 커스텀 loss 로직
        return custom_loss
```

## 🎨 Zero-shot Detection 예제들

```python
# 다양한 Zero-shot 프롬프트들
prompts = [
    "Find all cats and dogs",
    "Detect cars and motorcycles",
    "Locate people in the scene",
    "Find furniture like chairs and tables",
    "Detect electronic devices"
]

for prompt in prompts:
    boxes, scores, classes = model.detect_objects(
        tokenizer(prompt, return_tensors='pt')['input_ids'],
        image
    )
    print(f"{prompt}: {len(boxes[0])} objects detected")
```

## 📈 성능 최적화

- **Batch Processing**: 여러 이미지 동시 처리
- **Mixed Precision**: torch.cuda.amp 사용
- **Model Compilation**: torch.compile() 적용
- **KV Cache**: 텍스트 생성 시 캐시 활용

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🙏 감사의 말

- **HuggingFace**: 기본 VLM 아키텍처
- **SigLIP**: Vision encoder
- **SmolLM2**: Language model
- **YOLO**: Detection 아키텍처 영감

---

**nanoVLM**로 간단하면서도 강력한 Zero-shot Object Detection을 경험해보세요! 🎯✨
