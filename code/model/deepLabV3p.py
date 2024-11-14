import torch
import torch.nn as nn
import torch.nn.functional as F

# 반복적으로 나오는 구조를 쉽게 만들기 위해서 정의한 유틸리티 함수 입니다
def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride, 
                                   padding=(kernel_size // 2) * dilation, dilation=dilation, 
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
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
        self.conv1x1 = conv_block(in_ch, 256, 1, 1, 0)
        self.atrous_block1 = conv_block(in_ch, 256, 3, 1, padding=6, dilation=6)
        self.atrous_block2 = conv_block(in_ch, 256, 3, 1, padding=12, dilation=12)
        self.atrous_block3 = conv_block(in_ch, 256, 3, 1, padding=18, dilation=18)
        self.image_pooling = nn.AdaptiveAvgPool2d(1)
        self.image_conv = conv_block(in_ch, 256, 1, 1, 0)
        self.output_conv = conv_block(256 * 5, 256, 1, 1, 0)  # Added to match expected output channels

    def forward(self, x):
        x1 = self.conv1x1(x)
        x2 = self.atrous_block1(x)
        x3 = self.atrous_block2(x)
        x4 = self.atrous_block3(x)

        # Image-level feature
        image_feature = self.image_pooling(x)
        image_feature = self.image_conv(image_feature)
        image_feature = F.interpolate(image_feature, size=x.shape[2:], mode='bilinear', align_corners=True)

        # Concatenate all features
        out = torch.cat([x1, x2, x3, x4, image_feature], dim=1)
        
        # Adjust output channels to 256
        out = self.output_conv(out)
        
        return out





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
        self.backbone = Xception(in_channels)
        self.aspp = AtrousSpatialPyramidPooling(2048)
        self.decoder = Decoder(num_classes)

    def forward(self, x):
        # Backbone
        out, features = self.backbone(x)
        
        # ASPP
        out = self.aspp(out)
        
        # Decoder
        out = self.decoder(out, features)
        return out