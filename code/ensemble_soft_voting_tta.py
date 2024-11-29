import os
import sys
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import albumentations as A
import torch.nn.functional as F
import ttach as tta
import torch.nn as nn
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

import warnings
warnings.filterwarnings('ignore')
# 반복적으로 나오는 구조를 쉽게 만들기 위해서 정의한 유틸리티 함수 입니다
def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation=1):
        super().__init__()
        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2
        # TODO: depthwise conv - BN - pointwise conv로 구성된 레이어를 구현합니다
        self.depthwise = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_ch, bias=False)
        self.BN = nn.BatchNorm2d(num_features=in_ch)
        self.pointwise = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=1, bias=False) # bias=False는 권장 사항. 해도 되고 안 해도 됨.
        # 배치 정규화(Batch Normalization) 레이어가 뒤따르는 경우, 보통 bias=False를 설정합니다. 배치 정규화가 이미 편향과 유사한 역할을 수행하므로, 추가적인 bias 파라미터가 불필요해집니다.
        # bias=False를 설정하면 모델의 학습 파라미터 수가 줄어들고 학습 속도가 빨라지는 장점
        # bias=False가 필수는 아닙니다. 편향 값이 필요하다고 판단되면 bias=True로 설정해도 됩니다. 다만, 배치 정규화가 있을 때 bias를 추가하면 중복 효과가 발생하여 학습에 영향을 줄 수 있으므로 일반적으로는 bias=False를 사용하는 것이 권장
        
    def forward(self, x):
        # TODO: `__init__`에서 작성한 모듈을 이용하여 forward 함수를 작성하세요
        x = self.depthwise(x)
        x = self.BN(x)
        x = self.pointwise(x)
        return x


class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super().__init__()
        if in_ch != out_ch or stride !=1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.skip = None

        if exit_flow:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, in_ch, 3, 1, dilation),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch)
            ]
        else:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch)
            ]

        if not use_1st_relu:
            block = block[1:]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        output = self.block(x)
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        x = output + skip
        return x


class Xception(nn.Module):
    def __init__(self, in_channels):
        super(Xception, self).__init__()
        self.entry_block_1 = nn.Sequential(
            conv_block(in_channels, 32, 3, 2, 1),
            conv_block(32, 64, 3, 1, 1, relu=False),
            XceptionBlock(64, 128, 2, 1, use_1st_relu=False)
        )
        self.relu = nn.ReLU()
        self.entry_block_2 = nn.Sequential(
            XceptionBlock(128, 256, 2, 1),
            XceptionBlock(256, 728, 2, 1)
        )

        middle_block = [XceptionBlock(728, 728, 1, 1) for _ in range(16)]
        self.middle_block = nn.Sequential(*middle_block)

        self.exit_block = nn.Sequential(
            XceptionBlock(728, 1024, 1, 1, exit_flow=True),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1024, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 2048, 3, 1, 2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.entry_block_1(x)
        features = out
        out = self.entry_block_2(out)
        out = self.middle_block(out)
        out = self.exit_block(out)
        return out, features


class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        # TODO: ASPP를 구성하는 모듈들을 작성합니다
        #    중간 피처맵의 채널 사이즈는 256을 사용합니다
        self.aspp1 = conv_block(in_ch, 256, k_size=1, stride=1, padding=0, dilation=1)
        self.aspp2 = conv_block(in_ch, 256, k_size=3, stride=1, padding=6, dilation=6)
        self.aspp3 = conv_block(in_ch, 256, k_size=3, stride=1, padding=12, dilation=12)
        self.aspp4 = conv_block(in_ch, 256, k_size=3, stride=1, padding=18, dilation=18)
        self.aspp5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(in_channels=in_ch, out_channels=256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.output = nn.Sequential(
            nn.Conv2d(256 * 5, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def forward(self, x):
        # TODO: `__init__`에서 작성한 모듈을 이용하여 forward 함수를 작성하세요
        # 각 ASPP 분기를 통해 특징을 추출하고 결합합니다.
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        aspp5 = self.aspp5(x)
        aspp5 = F.interpolate(aspp5, size=aspp4.size()[2:], mode="bilinear", align_corners=True)
        
        # 채널을 결합하여 최종 ASPP 출력을 만듭니다.
        output = torch.cat((aspp1, aspp2, aspp3, aspp4, aspp5), dim=1)
        output = self.output(output)
        return output


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = conv_block(128, 48, 1, 1, 0)
        self.block2 = nn.Sequential(
            conv_block(48+256, 256, 3, 1, 1),
            conv_block(256, 256, 3, 1, 1),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x, features):
        features = self.block1(features)
        feature_size = (features.shape[-1], features.shape[-2])

        out = F.interpolate(x, size=feature_size, mode="bilinear", align_corners=True)
        out = torch.cat((features, out), dim=1)
        out = self.block2(out)
        return out


class DeepLabV3p(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # TODO: DeepLabV3+ 모델을 완성하기 위해 필요한 모듈들을 작성하세요
        #   상단에서 작성한 backbone Xception 모델과 ASPP 및 decoder를 사용합니다
        #   ASPP에서 중간 피처맵 사이즈로 256을 사용했다는 점을 이용해야 합니다
        # Backbone으로 Xception 네트워크 사용
        self.backbone = Xception(in_channels)
        
        # ASPP 모듈 정의 (backbone의 마지막 출력 채널 수인 2048을 입력으로 받음)
        self.aspp = AtrousSpatialPyramidPooling(2048)
        
        # Decoder 정의
        self.decoder = Decoder(num_classes)

    def forward(self, x):

        # TODO: `__init__`에서 작성한 모듈을 이용하여 forward 함수를 작성하세요
        # Backbone을 통해 특징 추출
        out, features = self.backbone(x)
        
        # ASPP 모듈 적용
        aspp_out = self.aspp(out)
        
        # Decoder를 통해 최종 출력 생성
        # ValueError: Target size (torch.Size([2, 29, 512, 512])) must be the same as input size (torch.Size([2, 29, 128, 128]))
        output = self.decoder(aspp_out, features)
        output = F.interpolate(output, scale_factor=4, mode="bilinear", align_corners=True)
        return output
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegFormer_B0(nn.Module):
    def __init__(self, num_classes=29):
        super(SegFormer_B0, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, image):
        outputs = self.model(pixel_values=image)
        upsampled_logits = nn.functional.interpolate(outputs.logits, size=image.shape[-1], mode="bilinear", align_corners=False)
        return upsampled_logits

class EnsembleDataset(Dataset):
    """
    Soft Voting을 위한 DataSet 클래스입니다. 이 클래스는 이미지를 로드하고 전처리하는 작업과
    구성 파일에서 지정된 변환을 적용하는 역할을 수행합니다.

    Args:
        fnames (set) : 로드할 이미지 파일 이름들의 set
        cfg (dict) : 이미지 루트 및 클래스 레이블 등 설정을 포함한 구성 객체
        tf_dict (dict) : 이미지에 적용할 Resize 변환들의 dict
    """    
    def __init__(self, fnames, cfg, tf_dict):
        self.fnames = np.array(sorted(fnames))
        self.image_root = cfg.image_root
        self.tf_dict = tf_dict
        self.ind2class = {i : v for i, v in enumerate(cfg.CLASSES)}

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, item):
        """
        지정된 인덱스에 해당하는 이미지를 로드하여 반환합니다.
        Args:
            item (int): 로드할 이미지의 index

        Returns:
            dict : "image", "image_name"을 키값으로 가지는 dict
        """        
        image_name = self.fnames[item]
        image_path = osp.join(self.image_root, image_name)
        image = cv2.imread(image_path)

        assert image is not None, f"{image_path} 해당 이미지를 찾지 못했습니다."
        
        image = image / 255.0
        return {"image" : image, "image_name" : image_name}

    def collate_fn(self, batch):
        """
        배치 데이터를 처리하는 커스텀 collate 함수입니다.

        Args:
            batch (list): __getitem__에서 반환된 데이터들의 list

        Returns:
            dict: 처리된 이미지들을 가지는 dict
            list: 이미지 이름으로 구성된 list
        """        
        images = [data['image'] for data in batch]
        image_names = [data['image_name'] for data in batch]
        inputs = {"images" : images}

        image_dict = self._apply_transforms(inputs)

        image_dict = {k : torch.from_numpy(v.transpose(0, 3, 1, 2)).float()
                      for k, v in image_dict.items()}
        
        for image_size, image_batch in image_dict.items():
            assert len(image_batch.shape) == 4, \
                f"collate_fn 내부에서 image_batch의 차원은 반드시 4차원이어야 합니다.\n \
                현재 shape : {image_batch.shape}"
            assert image_batch.shape == (len(batch), 3, image_size, image_size), \
                f"collate_fn 내부에서 image_batch의 shape은 ({len(batch)}, 3, {image_size}, {image_size})이어야 합니다.\n \
                현재 shape : {image_batch.shape}"

        return image_dict, image_names
    
    def _apply_transforms(self, inputs):
        """
        입력된 이미지에 변환을 적용합니다.

        Args:
            inputs (dict): 변환할 이미지를 포함하는 딕셔너리

        Returns:
            dict : 변환된 이미지들
        """        
        return {
            key: np.array(pipeline(**inputs)['images']) for key, pipeline in self.tf_dict.items()
        }


def encode_mask_to_rle(mask):
    # mask map으로 나오는 인퍼런스 결과를 RLE로 인코딩 합니다.
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def load_models(cfg, device):
    """
    구성 파일에 지정된 경로에서 모델을 로드합니다.

    Args:
        cfg (dict): 모델 경로가 포함된 설정 객체
        device (torch.device): 모델을 로드할 장치 (CPU or GPU)

    Returns:
        dict: 처리 이미지 크기별로 모델을 그룹화한 dict
        int: 로드된 모델의 총 개수
    """    
    model_dict = {}
    model_count = 0
    tta_transforms = tta.Compose(
        [
            tta.HorizontalFlip(),  # 0.0006 상승
            # tta.Add(values=[0, 3])
            # tta.Multiply(factors=[0.8, 1.0, 1.2, 1.4]),
            # tta.Scale(scales=[1.0, 1.2]),
            # tta.Add(value=[-10, 10]),
            # tta.VerticalFlip(),
            # tta.Rotate90(angles=[0, 90, 180, 270]),
        ]
    )

    print("\n======== Model Load ========")
    # inference 해야하는 이미지 크기 별로 모델 순차저장
    for key, paths in cfg.model_paths.items():
        if len(paths) == 0:
            continue
        model_dict[key] = []
        print(f"{key} image size 추론 모델 {len(paths)}개 불러오기 진행 시작")
        for path in paths:
            print(f"{osp.basename(path)} 모델을 불러오는 중입니다..", end="\t")
            model = torch.load(path).to(device)
            model.eval()
            tta_model = tta.SegmentationTTAWrapper(model, tta_transforms)
            model_dict[key].append(tta_model)
            model_count += 1
            print("불러오기 성공!")
        print()

    print(f"모델 총 {model_count}개 불러오기 성공!\n")
    return model_dict, model_count


def save_results(cfg, filename_and_class, rles):
    """
    추론 결과를 csv 파일로 저장합니다.

    Args:
        cfg (dict): 출력 설정을 포함하는 구성 객체
        filename_and_class (list): 파일 이름과 클래스 레이블이 포함된 list
        rles (list): RLE로 인코딩된 세크멘테이션 마스크들을 가진 list
    """    
    classes, filename = zip(*[x.split("_") for x in filename_and_class])
    image_name = [os.path.basename(f) for f in filename]

    df = pd.DataFrame({
        "image_name": image_name,
        "class": classes,
        "rle": rles,
    })

    print("\n======== Save Output ========")
    print(f"{cfg.save_dir} 폴더 내부에 {cfg.output_name}을 생성합니다..", end="\t")
    os.makedirs(cfg.save_dir, exist_ok=True)

    output_path = osp.join(cfg.save_dir, cfg.output_name)
    try:
        df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"{output_path}를 생성하는데 실패하였습니다.. : {e}")
        raise

    print(f"{osp.join(cfg.save_dir, cfg.output_name)} 생성 완료")



def soft_voting(cfg):
    """
    Soft Voting을 수행합니다. 여러 모델의 예측을 결합하여 최종 예측을 생성

    Args:
        cfg (dict): 설정을 포함하는 구성 객체
    """    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    fnames = {
        osp.relpath(osp.join(root, fname), start=cfg.image_root)
        for root, _, files in os.walk(cfg.image_root)
        for fname in files
        if osp.splitext(fname)[1].lower() == ".png"
    }

    tf_dict = {image_size : A.Resize(height=image_size, width=image_size) 
               for image_size, paths in cfg.model_paths.items() 
               if len(paths) != 0}
    
    print("\n======== PipeLine 생성 ========")
    for k, v in tf_dict.items():
        print(f"{k} 사이즈는 {v} pipeline으로 처리됩니다.")

    dataset = EnsembleDataset(fnames, cfg, tf_dict)
    
    data_loader = DataLoader(dataset=dataset,
                             batch_size=cfg.batch_size,
                             shuffle=False,
                             num_workers=cfg.num_workers,
                             drop_last=False,
                             collate_fn=dataset.collate_fn)

    model_dict, model_count = load_models(cfg, device)
    
    filename_and_class = []
    rles = []

    print("======== Soft Voting Start ========")
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="[Inference...]", disable=False) as pbar:
            for image_dict, image_names in data_loader:
                total_output = torch.zeros((cfg.batch_size, len(cfg.CLASSES), 2048, 2048)).to(device)
                for name, models in model_dict.items():
                    for model in models:
                        outputs = model(image_dict[name].to(device))
                        outputs = F.interpolate(outputs, size=(2048, 2048), mode="bilinear")
                        outputs = torch.sigmoid(outputs)
                        total_output += outputs
                        
                total_output /= model_count
                total_output = (total_output > cfg.threshold).detach().cpu().numpy()

                for output, image_name in zip(total_output, image_names):
                    for c, segm in enumerate(output):
                        rle = encode_mask_to_rle(segm)
                        rles.append(rle)
                        filename_and_class.append(f"{dataset.ind2class[c]}_{image_name}")
                
                pbar.update(1)

    save_results(cfg, filename_and_class, rles)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="utils/soft_voting_setting.yaml")

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        cfg = OmegaConf.load(f)

    if cfg.root_path not in sys.path:
        sys.path.append(cfg.root_path)
    
    soft_voting(cfg)