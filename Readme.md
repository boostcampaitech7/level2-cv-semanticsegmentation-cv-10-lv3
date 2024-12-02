# Hand Bone Segmentation Project
#
## Overview

손 뼈 분할(segmentation)은 질병 진단, 수술 계획, 의료 교육 등 다양한 의료 분야에서 중요한 역할을 합니다. 본 프로젝트는 고해상도(2048x2048) X-ray 이미지에서 29개의 손 뼈 레이블을 분할하는 작업에 초점을 맞췄습니다. 데이터 분석(EDA)부터 다양한 증강 기법, 슬라이딩 윈도우, 이미지 크롭 및 앙상블 기법까지 포괄적인 접근을 통해 성능을 개선했습니다.

#
## EDA

### 데이터셋 개요

- **해상도**: 2048x2048 크기의 고해상도 X-ray 이미지.
- **레이블**: 각 이미지에는 29개의 손 뼈 레이블 포함.
- **구성**: 모든 데이터셋이 좌우 손 쌍으로 구성.

### 특징

- `ID363`: 악세서리(반지)를 착용한 데이터.
- `ID274~321`: 일반적인 포즈와 다른 자세를 취한 데이터.
- 손등 부분에 겹치는 클래스가 다수 존재하며, 특히 Trapezoid, Pisiform 클래스의 정확도가 낮음 (0.88).

#
## Methods

### Augmentation

다양한 증강 기법을 적용하여 모델 성능을 향상시켰습니다.

- **CLAHE**: 대조를 강화하여 뼈 사이의 경계 인식 개선.
- **ElasticTransform**: 뼈의 다양한 휘어짐에 대응.
- **GaussianBlur**: 모델의 일반화 성능 향상.
- **Erode**: 경계를 축소시켜 손등 뼈의 특징 강화.
- **Gradient**: 테두리 경계 강화를 통해 분할 성능 향상.


### 학습 전략

- **Sliding Window**  
  이미지를 512x512 타일로 분할하여 학습.  
  메모리 부족으로 나눠서 학습했지만, 성능이 낮아졌음.
- **이미지 Crop**  
  손등 영역을 중심으로 학습하여 겹치는 뼈의 분할 성능을 개선. 기존보다 0.0011 상승.
- **Resize**  
  이미지 크기를 512에서 1024 또는 2048로 조정하여 학습 진행.

#
## Modeling

### 모델 선택

- SegFormer, Mask2Former, ConvNext, Unet++, DeepLabV3+ 등 다양한 모델 실험.
- Unet++ 및 DeepLabV3+를 중심으로 데이터 증강 및 학습 전략 최적화.

### 손실 함수

- **BCE_Dice_loss**: BCE와 Dice Loss를 조합하여 학습.

### 최적화

- 기존 RMSprop보다 **AdamW**를 사용하여 0.24의 성능 향상.
```python
optim.AdamW(params=model.parameters(), lr=LR, weight_decay=1e-5)
```
- Cosine Annealing Scheduler로 학습률을 진동시키며 최적화.
```python
CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-8)
```

#
## Post-Processing

### Test-Time Augmentation (TTA)

- **HorizontalFlip()**을 적용하여 성능이 0.0006 향상됨.

#
## Ensemble

### 앙상블 전략

- **Hard Voting**
- **Soft Voting**
- **Class-wise 앙상블**

### K-Fold 앙상블

- 각 모델을 기반으로 5-Fold 실험을 수행.
- Hard Voting 및 Soft Voting 기법을 통해 성능 변동성을 줄이고 일관된 성능을 확보.

#
## Conclusion

본 프로젝트는 고해상도 X-ray 이미지에서 손 뼈를 분할하기 위한 효과적인 방법을 제안했습니다. 데이터 분석, 증강 기법, 학습 전략, 앙상블 기술을 결합하여 분할 성능을 크게 개선할 수 있었습니다. 특히 앙상블과 이미지 Crop 전략은 손바닥 뼈와 같이 분할이 어려운 영역에서 유의미한 개선을 보여주었습니다.

