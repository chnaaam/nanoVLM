import json
import os
import tempfile
from dataclasses import asdict
from typing import Optional, List, Tuple


from models.utils import top_k_top_p_filtering
from models.vision_transformer import ViT
from models.language_model import LanguageModel
from models.modality_projector import ModalityProjector
from models.region_model import RegionModel
from models.detection_losses import DetectionLossManager
from models.config import VLMConfig

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model, save_model

class VisionLanguageDetectionModel(nn.Module):
    """Zero-shot Object Detection을 위한 Vision-Language 모델"""
    def __init__(self, cfg: VLMConfig, load_backbone=True):
        super().__init__()
        self.cfg = cfg
        if load_backbone:
            print("Loading from backbone weights")
            self.vision_encoder = ViT.from_pretrained(cfg)
            self.decoder = LanguageModel.from_pretrained(cfg)
        else:
            self.vision_encoder = ViT(cfg)
            self.decoder = LanguageModel(cfg)

        # Freeze the vision encoder
        for param in self.vision_encoder.parameters():
            param.requires_grad = False

        # Freeze the language model
        for param in self.decoder.parameters():
            param.requires_grad = False

        self.MP = ModalityProjector(cfg)
        self.region_model = RegionModel(cfg)
        self.load_backbone = load_backbone
        
        # Detection loss manager 초기화
        self.detection_loss_manager = DetectionLossManager(cfg)

    def forward(self, input_ids, image, attention_mask=None, targets=None, detection_targets=None):
        """
        Args:
            input_ids: 텍스트 토큰 ID
            image: 입력 이미지
            attention_mask: 어텐션 마스크
            targets: 언어 모델링 타겟 (optional)
            detection_targets: Object detection 타겟 (optional)
        """
        # 1. Vision encoding
        image_embd = self.vision_encoder(image)  # [B, H*W, D]
        vision_features_for_detection = image_embd.clone()  # Detection용 복사본
        
        # 2. Modality projection for language model
        image_embd = self.MP(image_embd)

        # 3. Language processing
        token_embd = self.decoder.token_embedding(input_ids)
        combined_embd = torch.cat((image_embd, token_embd), dim=1)
        
        # Adjust attention mask to account for image tokens
        if attention_mask is not None:
            batch_size = image_embd.size(0)
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)

        language_outputs, _ = self.decoder(combined_embd, attention_mask=attention_mask)
        
        # 4. Object Detection
        # 텍스트 임베딩을 detection에 활용
        text_embeddings = language_outputs[:, image_embd.size(1):, :]  # 텍스트 부분만 추출
        
        detections, class_embeddings = self.region_model(
            vision_features_for_detection, 
            text_embeddings=text_embeddings
        )

        # 5. Loss computation
        total_loss = 0.0
        losses = {}
        
        # Language modeling loss
        if targets is not None:
            language_logits = self.decoder.head(language_outputs)
            language_logits = language_logits[:, image_embd.size(1):, :]
            lm_loss = F.cross_entropy(
                language_logits.reshape(-1, language_logits.size(-1)), 
                targets.reshape(-1), 
                ignore_index=-100
            )
            losses['language_loss'] = lm_loss
            total_loss += lm_loss
        
        # Detection loss
        if detection_targets is not None:
            detection_loss = self._compute_detection_loss(detections, detection_targets)
            losses['detection_loss'] = detection_loss
            total_loss += detection_loss

        return {
            'detections': detections,
            'class_embeddings': class_embeddings,
            'language_outputs': language_outputs,
            'losses': losses,
            'total_loss': total_loss if targets is not None or detection_targets is not None else None
        }

    def _compute_detection_loss(self, detections, targets, loss_type='yolo'):
        """개선된 YOLO 스타일의 detection loss 계산"""
        return self.detection_loss_manager.compute_total_loss(
            detections, targets, loss_type=loss_type
        )

    @torch.inference_mode()
    def detect_objects(self, input_ids, image, attention_mask=None, conf_threshold=None, nms_threshold=None):
        """
        Zero-shot object detection을 수행합니다.
        
        Args:
            input_ids: 텍스트 프롬프트 (예: "Find all cats and dogs")
            image: 입력 이미지
            attention_mask: 어텐션 마스크
            conf_threshold: Confidence threshold (optional)
            nms_threshold: NMS threshold (optional)
        
        Returns:
            boxes: 검출된 bounding box들
            scores: Confidence scores
            classes: Class indices
        """
        # Threshold 설정
        if conf_threshold is not None:
            self.region_model.conf_threshold = conf_threshold
        if nms_threshold is not None:
            self.region_model.nms_threshold = nms_threshold
        
        # Forward pass
        outputs = self.forward(input_ids, image, attention_mask)
        detections = outputs['detections']
        
        # Detection 결과 디코딩
        boxes, scores, classes = self.region_model.decode_predictions(
            detections, 
            img_size=self.cfg.vit_img_size
        )
        
        return boxes, scores, classes

    @torch.inference_mode()
    def generate_with_detection(self, input_ids, image, attention_mask=None, max_new_tokens=50, 
                               top_k=50, top_p=0.9, temperature=0.5, greedy=False,
                               return_detections=True):
        """
        텍스트 생성과 동시에 객체 탐지를 수행합니다.
        
        Args:
            input_ids: 텍스트 프롬프트
            image: 입력 이미지
            max_new_tokens: 생성할 최대 토큰 수
            return_detections: Detection 결과도 함께 반환할지 여부
        
        Returns:
            generated_tokens: 생성된 토큰들
            detections: 검출 결과 (optional)
        """
        # 1. Process image
        image_embd = self.vision_encoder(image)
        vision_features_for_detection = image_embd.clone()
        image_embd = self.MP(image_embd)

        # 2. Embed initial text prompt tokens
        prompt_token_embeds = self.decoder.token_embedding(input_ids)
        initial_combined_embeds = torch.cat((image_embd, prompt_token_embeds), dim=1)
        current_total_seq_len = initial_combined_embeds.size(1)

        batch_size = image_embd.size(0)
        if attention_mask is not None:
            img_seq_len = image_embd.size(1)
            image_attention_mask = torch.ones((batch_size, img_seq_len), device=attention_mask.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat((image_attention_mask, attention_mask), dim=1)
        
        # 3. Multimodal Prefill Phase
        prefill_output, kv_cache_list = self.decoder(
            initial_combined_embeds,
            attention_mask=attention_mask,
            kv_cache=None,
            start_pos=0
        )
        
        last_token_output_from_prefill = prefill_output[:, -1, :] 
        
        if not self.decoder.lm_use_tokens:
            current_logits = self.decoder.head(last_token_output_from_prefill) 
        else:
            current_logits = last_token_output_from_prefill 

        # Store newly generated token IDs
        newly_generated_ids_list = []

        # 4. Decode Phase
        for _ in range(max_new_tokens):
            if greedy:
                next_token_id = torch.argmax(current_logits, dim=-1, keepdim=True)
            else:
                filtered_logits = top_k_top_p_filtering(current_logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(filtered_logits / temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            newly_generated_ids_list.append(next_token_id)
            
            next_token_embed = self.decoder.token_embedding(next_token_id)
            current_token_start_pos = current_total_seq_len
            current_total_seq_len += 1

            if attention_mask is not None:
                attention_mask = torch.cat((attention_mask, torch.ones((batch_size, 1), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)

            decode_step_output, kv_cache_list = self.decoder(
                next_token_embed,
                attention_mask=attention_mask,
                kv_cache=kv_cache_list,
                start_pos=current_token_start_pos
            )
      
            last_token_output = decode_step_output[:, -1, :] 
            
            if not self.decoder.lm_use_tokens:
                current_logits = self.decoder.head(last_token_output)
            else:
                current_logits = last_token_output

        # 5. Object Detection (if requested)
        detections_result = None
        if return_detections:
            # 전체 텍스트 임베딩 사용
            full_text_embeddings = torch.cat([prefill_output[:, image_embd.size(1):, :]] + 
                                           [self.decoder.token_embedding(torch.cat(newly_generated_ids_list, dim=1))], dim=1)
            
            detections, class_embeddings = self.region_model(
                vision_features_for_detection,
                text_embeddings=full_text_embeddings
            )
            
            boxes, scores, classes = self.region_model.decode_predictions(
                detections,
                img_size=self.cfg.vit_img_size
            )
            
            detections_result = {
                'boxes': boxes,
                'scores': scores,
                'classes': classes,
                'detections': detections,
                'class_embeddings': class_embeddings
            }

        generated_tokens = torch.cat(newly_generated_ids_list, dim=1) if newly_generated_ids_list else torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
        
        if return_detections:
            return generated_tokens, detections_result
        else:
            return generated_tokens

    @classmethod
    def from_pretrained(
        cls, repo_id_or_path: str, *, revision: Optional[str] = None
    ) -> "VisionLanguageDetectionModel":
        """
        Load a VisionLanguageDetectionModel from a local directory or a repo on the Hugging Face Hub.
        """
        # If local folder exists => load from there
        if os.path.exists(repo_id_or_path):
            config_path = os.path.join(repo_id_or_path, "config.json")
            weights_path = os.path.join(repo_id_or_path, "model.safetensors")

            if not os.path.exists(config_path):
                raise ValueError(
                    f"Config file not found at {config_path}. Please provide a valid path."
                )
            if not os.path.exists(weights_path):
                raise ValueError(
                    f"Weights file not found at {weights_path}. Please provide a valid path."
                )
        else:
            from huggingface_hub import hf_hub_download

            config_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="config.json", revision=revision
            )
            weights_path = hf_hub_download(
                repo_id=repo_id_or_path, filename="model.safetensors", revision=revision
            )

        # Load config
        with open(config_path, "r") as f:
            cfg = VLMConfig(**json.load(f))

        # Initialize model without loading the backbone
        model = cls(cfg, load_backbone=False)

        # Load safetensors weights
        load_model(model, weights_path)

        return model

    def save_pretrained(self, save_directory: str) -> None:
        """
        Save the model and configuration to a directory.
        """
        os.makedirs(save_directory, exist_ok=True)

        # Save config
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            f.write(json.dumps(asdict(self.cfg), indent=4))

        # Save weights as safetensors
        save_model(self, os.path.join(save_directory, "model.safetensors"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> None:
        """
        Push the model and configuration to the Hugging Face Hub.
        """
        from huggingface_hub import create_repo, upload_folder

        repo_url = create_repo(repo_id=repo_id, private=private, exist_ok=True)
        repo_id = repo_url.repo_id
        print("Created repo: ", repo_url)

        with tempfile.TemporaryDirectory() as save_path:
            self.save_pretrained(save_path)

            with open(os.path.join(save_path, "README.md"), "w") as f:
                f.write(MODEL_CARD_TEMPLATE.format(repo_id=repo_id))

            return upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=save_path,
                commit_message="Upload nanoVLM Detection Model using push_to_hub",
            )


# 기존 VisionLanguageModel을 VisionLanguageDetectionModel의 alias로 유지 (하위 호환성)
VisionLanguageModel = VisionLanguageDetectionModel


MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
library_name: nanovlm
license: mit
pipeline_tag: zero-shot-object-detection
tags:
  - vision-language
  - multimodal
  - object-detection
  - zero-shot
  - research
---

**nanoVLM Detection** is a minimal and lightweight Zero-shot Object Detection model that combines Vision-Language understanding with YOLO-style object detection. Built using pure PyTorch, it integrates a ViT-based image encoder (SigLIP-B/16-224-85M) with a lightweight causal language model (SmolLM2-135M) and a YOLO-style detection head.

This model can perform zero-shot object detection by understanding natural language descriptions of objects to detect.

For more information, check out the base model on https://huggingface.co/lusxvr/nanoVLM-222M.

**Usage:**

Clone the nanoVLM repository: https://github.com/huggingface/nanoVLM.
Follow the install instructions and run the following code:

```python
from models.vision_language_model import VisionLanguageDetectionModel

model = VisionLanguageDetectionModel.from_pretrained("{repo_id}")

# For object detection
boxes, scores, classes = model.detect_objects(input_ids, image)

# For text generation with detection
generated_text, detections = model.generate_with_detection(input_ids, image)
```
"""
