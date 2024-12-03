# Hand Bone Segmentation Project


## Overview

뼈는 우리 몸의 구조와 기능에 중요한 영향을 미치기 때문에, 정확한 뼈 분할은 의료 진단 및 치료 계획을 개발하는 데 필수적이다. Bone Segmentation은 인공지능 분야에서 중요한 응용 분야 중 하나로, 특히, 딥러닝 기술을 이용한 뼈 Segmentation은 많은 연구가 이루어지고 있으며, 다양한 목적으로 도움을 줄 수 있다.
- **질병 진단** :  손 뼈 변형, 골절 등 이상 탐지
- **수술 계획** : 뼈 구조 분석을 통한 수술 방식 결정
- **의료 장비 제작** : 인공 관절, 임플란트 등의 맞춤 제작
- **의료 교육** : 뼈 구조 학습 및 수술 시뮬레이션

![image](https://github.com/user-attachments/assets/a4250715-2b38-44d5-8663-f07e753d85ef)

## Member
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/Ai-BT"><img height="110px" src="https://avatars.githubusercontent.com/u/97381138?v=4"/></a>
            <br />
            <a href="https://github.com/Ai-BT"><strong>김대환</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/SkyBlue-boy"><img height="110px"  src="https://avatars.githubusercontent.com/u/63849988?v=4"/></a>
              <br />
              <a href="https://github.com/SkyBlue-boy"><strong>박윤준</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/sweetie-orange"><img height="110px"  src="https://avatars.githubusercontent.com/u/97962649?v=4"/></a>
              <br />
              <a href="https://github.com/sweetie-orange"><strong>김현진</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/0seoYun"><img height="110px"  src="https://avatars.githubusercontent.com/u/102219161?v=4"/></a>
              <br />
              <a href="https://github.com/0seoYun"><strong>윤영서</strong></a>
              <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/jiy0-0nv"><img height="110px"  src="https://avatars.githubusercontent.com/u/128347728?v=4"/></a>
              <br />
              <a href="https://github.com/jiy0-0nv"><strong>정지윤</strong></a>
              <br />
        </td>
    </tr>
</table>  


## EDA

### 데이터셋 개요

- **구성** : 총 2048x2048 해상도의 X-ray 이미지로 구성된 데이터셋, 양손 모두 포함
- **클래스** : 29개의 손 뼈 레이블 포함
- **구조** : 모든 데이터가 좌우 손으로 쌍을 이룸

### 특징

- ID363 데이터는 악세서리(반지)를 착용하고 있음
- 일부 데이터(ID274~321)는 일반적인 손 자세와 다른 포즈를 취한 데이터
- 손등 뼈(Trapezoid, Pisiform 등) 간 겹침이 발생하며, 해당 클래스의 정확도가 다른 클래스보다 낮음(Val acc **0.90**)
- 손가락 끝 부분 (f1, f4, f8, ,f12, f16) 성능이 낮음 (Val acc **0.90**)

### 문제 정의

- 손이 휘어진 데이터를 대응하기 위해 Rotate 증강 (limit=15,30) 으로 적용
- 손등 부분을 Crop하여 모델이 해당 영역을 집중적으로 학습할 수 있도록 데이터 전처리 및 학습 과정 설계
- 클래스 간 경계 개선을 위해 ElasticTransform 추가 증강 적용
- 작고 겹치는 클래스의 성능 향상을 위해 이미지 크기를 512에서 1024, 2048로 점진적으로 확대하며 학습을 진행


## Methods

### Model Selection

| 모델 | Image size | Epoch | Time | Score |
| --- | --- | --- | --- | --- |
| Unet | 512 | 100 | 3.5 h | 93.72 % |
| Unet++ | 512 | 100 | 4.3 h | 95.01 % |

위의 2개 모델 간 성능 차이는 약 0.2%로 차이가 있었으며, 학습 시간 역시 약 **40분 차이**에 불과하여 **Base 모델**로 UNet++을 선정하고 다양한 실험을 진행했습니다. 반면, 다른 모델(Segformer, Convnext, DeepLab3v+)들은 학습 시간이 **6~7시간** 정도 소요되어 Base 모델에서 제외하였다.

### Augmentation
Base UNet++ 모델을 기준으로 Augmentation 기법을 하나씩 변경하며 단계적으로 실험
| **Augmentation**         | **Condition**                                                            | **Epoch** | **Score**   | **Result**   |
|---------------------------|-------------------------------------------------------------------------|----------|-------------|--------------|
| **Base**                 | Base                                                                   | 100      | 95.08 %     | Base         |
| **HorizontalFlip**        | p=0.5                                                                 | 100      | 95.33 %     | 0.25 % 상승  |
| **GaussianBlur**          | blur_limit=(3, 7), sigma_limit=(0.1, 2), p=0.5                         | 100      | 95.61 %     | 0.32 % 상승  |
| **ElasticTransform**      | alpha=1, sigma=50, p=0.5                                              | 100      | 95.43 %     | 0.35 % 상승  |
| **GridDistortion**        | ratio=0.4, random_offset=False, holes_number_x=12, holes_number_y=12, p=0.2 | 100      | 95.42 %     | 0.34 % 상승  |
| **Rotate**                | limit=30, p=0.3                                                       | 100      | 95.28 %     | 0.20 % 상승  |
| **CLAHE**                 | clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5                        | 100      | 95.33 %     | 0.25 % 상승  |
| **RandomBrightnessContrast** | brightness_limit=(0.0, 0.3), contrast_limit=0.3, p=0.5               | 100      | 95.10 %     | 0.02 % 상승  |
| **GridDropout**           | distort_limit=0.2, p=0.4                                              | 100      | 95.10 %     | 0.02 % 상승  |

최종적으로 성능이 상승한 Augmentation 기법을 채택한 뒤, 추가적인 실험을 이어갔다.

EDA 과정에서 발견된 문제점인 **손등 뼈 영역의 성능 저하**를 개선하기 위해 **CenterCrop**, **RandomCrop**, **RandomResizedCrop** 등의 Crop 기반 기법을 사용하여 학습을 시도했다. 그러나, 이러한 방식은 오히려 성능이 떨어지는 결과를 보였다.

이에 따라, 새로운 접근으로 **Sliding Window**와 **Image Crop**을 적용하여 학습을 진행했다. Sliding Window는 이미지를 타일로 분할해 학습하고, Image Crop은 손등 중심으로 특정 영역을 잘라 학습에 집중하도록 하여 성능 향상을 도모했다.

### Sliding Window, Image Crop 학습 실험

기존의 Crop 기반 Augmentation은 성능이 저하되는 결과를 보였고, **손등 뼈의 정확도 향상**을 목표로 새로운 방식의 Crop 기법을 도입하여 실험을 진행했다.

**Sliding Window**

1. Datasets을 만들 때, 이미지를 불러와 Tile(512x512, Stride 256)을 생성
2. 이 중 학습할 수 있는 부분이 10% 미만일 경우 제외
3. 메모리 한계(OOM) 문제를 해결하기 위해  이미지를 3개의 파트로 분할하여 학습
4. Inference 단계에세도 동일하게 이미지를 Tile화 하여 예측 결과를 출력

결과

| Model | result |
| --- | --- |
| BaseModel | 94.09 % |
| SlidingWindow | 86.16 % |

성능이 크게 떨어진 원인으로는 **이미지 간 차이가 큰 데이터 특성**, **충분한 Epoch 학습 부족**, 그리고 **이미지를 3파트로 나누어 학습한 방식의 한계**가 주요 요인으로 분석된다.

**Image Crop 학습 후 결과 적용**

1. 손등 뼈의 위치를 정확히 식별하여 해당 영역의 좌표를 구함
2. 손등 뼈가 포함된 범위를 기준으로 Train 이미지를 **512x512 크기**로 잘라 학습 데이터 생성
3. 손등 뼈 영역만을 집중적으로 학습하여 성능 향상 도모
4. Pseudo Labeling 결과를 활용하여 Test 이미지에서도 손등 영역만 나오도록 **512x512 크기로 Crop**
5. Inference를 한 후 학습된 손등 뼈 데이터를 기반으로 기존 모델의 손등 뼈 예측 결과를 대체

**적용 모델**
UNet++, SegFormer B4, ConvNext XL에 Augmentation을 결합하여 최종 학습 및 평가를 진행.

결과

| Model | result |
| --- | --- |
| BaseModel | 97.45 % |
| Crop Change | 97.54 % |

Crop 후 학습한 이미지 결과로 대체한 결과 성능이 증가하였음을 알 수 있었다.

### Pseudo labeling

Pseudo Labeling을 활용해 **레이블이 없는 데이터를 학습 데이터로 추가**하여 모델의 일반화 성능과 정확도를 0.27% 향상시켰다.

| Model | result |
| --- | --- |
| BaseModel | 97.00 % |
| PseudoModel | 97.27 % |


## Modeling

### Resolution

**메모리 문제 해결**

- AMP(FP16 + FP32)를 활용해 CUDA 메모리 제한 문제를 해결
- AMP는 모델 성능 저하 없이 연산 속도를 높이고 메모리 효율성을 개선하여 안정적인 학습을 가능하게 함

**해상도에 따른 성능 향상**

- 해상도를 높일수록 성능이 크게 향상됨을 확인
- **SegFormer B4, ConvNext** 등 무거운 모델은 2048 해상도에서 **OOM 발생**, 1024 해상도로 제한
- **SegFormer B0**와 같은 가벼운 모델은 2048 해상도에서 더 우수한 성능을 보임

**최종 학습 전략**

- **2048 해상도 학습 가능 모델**: 2048 해상도로 학습
- **OOM 발생 모델**: 1024 해상도로 학습

**결과 요약**

- **2048 해상도**를 적용한 가벼운 모델이 성능 향상에 가장 중요한 요소로 확인
- **DeepLabV3+** 성능:

| Model | Image size | Result |
| --- | --- | --- |
| DeepLabV3+ | 512 | 95.50 % |
| DeepLabV3+ | 1024 | 96.63 % |
| DeepLabV3+ | 2048 | 97.07 % |

## Loss

**문제**

- 큰 뼈에 비해 **작은 뼈**를 정확히 예측하지 못하는 문제가 지속됨.
- **BCE Loss**는 픽셀 단위 분류에 유용하지만 작은 객체 탐지에는 한계가 있음.

**가설 및 접근**

- **Dice Loss**: 정밀도와 재현율 최적화.
- **Focal Loss**: 어려운 샘플(작은 객체)에 가중치를 부여하여 탐지 성능 향상 기대.
- **Focal+Dice 조합**: 두 손실 함수의 균형을 맞춰 성능 개선 도모.
- **Focal4+Dice6 조정**: Focal Loss의 과도한 가중치를 완화하여 다른 픽셀 학습 약화 문제 해결.

결과 (UNet3+ 기준)

| BCE | 95.42 % |
| --- | --- |
| BCE+Dice | 95.35 % |
| Focal+Dice | 95.32 % |
| Focal4+Dice6 | 95.57 % |

결과(DeepLabV3+ 기준)

| (Baseline)BCE | 95.38 % |
| --- | --- |
| BCE+Dice(bce_weight=0.5) | 95.50 % |
| Focal+Dice | 95.23 % |
| Focal4+Dice6 | 95.21 %  |

결론 및 최종 선택

- **BCE+Dice**: **DeeplabV3+, ConvNext, SegFormer**에 적용
- **Focal4+Dice6**: **UNet2+, UNet3+**에 적용

작은 객체 탐지 성능 향상을 위해 Focal Loss와 Dice Loss의 비율 조정이 효과적임


## Scheduler

**Cosine Annealing Learning Rate Scheduler 채택**

- 학습 초기: 높은 학습률로 빠르게 수렴
- 학습 후반: 학습률을 점진적으로 감소시켜 안정적 학습과 과적합 방지
- **장점**: 자연스러운 감소폭으로 안정적 학습 환경 제공

**결과**

- **UNet++** 기준, 베이스라인 대비 0.**01 % 성능 향상** 확인
- Cosine Annealing Scheduler를 최종적으로 채택


## Optimizer

AdamW를 최종적으로 채택하였다. 과적합 방지에 더 효과적인 AdamW를 사용하면 일반화 성능이 향상하여 성능이 더 좋다고 판단했다. 

결과(UNet++ 기준 0.24% 상승)

| RMSprop | 95.29 % |
| --- | --- |
| AdamW | 95.53 % |



## Post-Processing

CRF (Conditional Random Field)

CRF는 segmentation 모델의 출력(확률 맵)을 기반으로 공간 및 색상 정보를 활용하여 **픽셀 간 관계를 최적화**하고, **세밀한 분류 성능**을 개선하는 후처리 기법이다. 

**실험 결과**

1. **기본 파라미터**:
    - 파일 용량이 지나치게 커져 채점 불가능.
    - 원인: **과도한 smoothing 효과**로 데이터가 과도하게 변형됨.
2. **하이퍼파라미터 조정**:
    - 성능이 오히려 감소.
    - 원인 분석:
        1. **하이퍼파라미터 의존성**: 부적절한 값 설정으로 세부 정보 손실 발생.
        2. **부적합한 smoothing**: 복잡한 경계를 지나치게 단순화하여 작은 객체나 얇은 영역이 제거되거나 병합되는 오류 발생.
        

CRF는 픽셀 단위의 분류 오류와 계단 현상을 해결하려는 목적에서 도입되었으나, 적절한 하이퍼파라미터 조정이 부족하여 성능 개선에 실패했다. 특히, 과도한 smoothing으로 인해 성능 저하와 데이터 처리 문제를 야기했다.



## TTA

다양한 TTA를 실험했지만 HorizontalFlip 만이 기존보다 dice가 0.06 % 상승하는 유의미한 결과를 보여주어 최종 TTA에는 HorizontalFlip만을 적용하였다.

## Ensemble

모델 실험에서 성능이 우수했던 모델들을 선정하여 **K-Fold**를 진행한 뒤, 이를 기반으로 **Soft Voting**, **Hard Voting**, **Classwise Ensemble** 등 다양한 앙상블 방법을 실험했다.

최종적으로 가장 좋은 성능을 보인 방법은 **Classwise Ensemble**로, 각 클래스에서 가장 우수한 예측을 수행한 모델의 결과를 사용해 추론하는 방식이다.

특히, Crop한 이미지를 활용해 손등 뼈를 중심으로 학습한 데이터를 기반으로 Classwise Ensemble을 진행하여 손등 뼈 부분의 정확도를 효과적으로 향상시켰다.

KFOLD

- **Convnext Xlarge 1024 + 5fold**
- **Segformer b4 1024 + 5fold**
- **DeeplabV3p 2048 + 4fold**
- **Segformer b0 2048 + 4fold**

Final Ensemble

최종 Clsswise Ensemble 97.54% 달성했다.

## Conclusion

X-ray 손 뼈 분할의 주요 과제를 해결하기 위해 다양한 증강 기법, 모델 아키텍처, 앙상블 기법을 적용했다. 특히, 원본 이미지 학습 모델 기반 Classwise Ensemble 기법은 손등 뼈와 같은 어려운 영역에서 성능 향상을 이끌었다. 또한, SOTA 같은 무거운 모델을 낮은 해상도로 학습하는 것보다 가벼운 모델을 높은 해상도로 학습하는 것이 본 대회 성능 향상에 매우 중요했음을 알게 되었다. 향후 연구에서는 도메인 지식을 활용한 더 정교한 전처리 및 후처리 기법을 적용하여 더욱 높은 성능을 목표로 할 것이다.

## 최종 성과

- Public Dice Score: 97.56%
- Private Dice Score: 97.64 %


## Reference

- [1] Code for ICASSP 2020 paper ‘UNet 3+: A full-scale connected unet for medical image segmentation https://github.com/ZJUGiveLab/UNet-Version/tree/master
- [2] Python wrapper to Philipp Krähenbühl's dense (fully connected) CRFs with gaussian edge potentials https://github.com/lucasb-eyer/pydensecrf
- [3] Enze Xie et al., "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" arXiv:2105.15203v3 [[cs.CV](http://cs.cv/)] 28 Oct 2021
- [4] Semantic Segmentation on ADE20K SOTA https://paperswithcode.com/sota/semantic-segmentation-on-ade20k