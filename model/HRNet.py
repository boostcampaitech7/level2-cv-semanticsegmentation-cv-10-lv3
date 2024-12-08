# torch
import torch
import torch.nn as nn
import torch.nn.functional as F


class SteamBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
    def forward(self,inputs):
        return self.block(inputs)

class Stage01Block(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,256,kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
        )
        if in_channels == 64:
            self.identity_block=nn.Sequential(nn.Conv2d(in_channels, 256, kernel_size=1, bias=False), 
                                nn.BatchNorm2d(256))
            self.relu=nn.ReLU()
            self.inchannels = in_channels

    def forward(self, inputs):
        identity=inputs
        out=self.block(inputs)

        if self.in_channels == 64:
            identity = self.identity_block(identity)
        out+=identity
            
        return self.relu(out)
    
class Stage01StreamGenerateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_res_block = nn.Sequential(
            nn.Conv2d(256,48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.medium_res_block = nn.Sequential(
            nn.Conv2d(256,96,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
    def forward(self,inputs):
        out_high = self.high_res_block(inputs)
        out_medium = self.medium_res_block(inputs)
        return out_high, out_medium

class StageBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
        )
        self.relu=nn.ReLU()
    
    def forward(self, inputs):
        identity = inputs
        out = self.block(inputs)
        out += identity
        out = self.relu(out)
        return out
    
class Stage02(nn.Module):
    def __init__(self):
        super().__init__()
        high_res_blocks = [StageBlock(48) for _ in range(4)]
        medium_res_blocks =[StageBlock(96) for _ in range(4)]

        self.high_res_blocks = nn.Sequential(*high_res_blocks)
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks)

    def forward(self, inputs_high, inputs_medium):
        out_high = self.high_res_blocks(inputs_high)
        out_medium = self.medium_res_blocks(inputs_medium)
        return out_high, out_medium
    
class Stage02Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96)
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96,48,kernel_size=1,bias=False),
            nn.BatchNorm2d(48)
        )
        self.relu = nn.ReLU()
    
    def forward(self, inputs_high, inputs_medium):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        
        med2high = F.interpolate(
            inputs_medium, size=high_size, mode="bilinear", align_corners=True
        )
        med2high=self.medium_to_high(med2high)
        high2med=self.high_to_medium(inputs_high)

        out_high=inputs_high+med2high
        out_medium= inputs_medium+high2med

        out_high= self.relu(out_high)
        out_medium= self.relu(out_medium)
        return out_high, out_medium
    
class StreamGenerateBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels*2),
            nn.ReLU()
        )
    def forward(self,inputs):
        return self.block(inputs)

class Stage03(nn.Module):
    def __init__(self):
        super().__init__()
        high_res_blocks = [StageBlock(48) for _ in range(3)]
        medium_res_blocks =[StageBlock(96) for _ in range(3)]
        low_res_blocks =[StageBlock(192) for _ in range(3)]

        self.high_res_blocks = nn.Sequential(*high_res_blocks)
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks)
        self.low_res_blocks = nn.Sequential(*low_res_blocks)

    def forward(self, inputs_high, inputs_medium,inputs_low):
        out_high = self.high_res_blocks(inputs_high)
        out_medium = self.medium_res_blocks(inputs_medium)
        out_low = self.low_res_blocks(inputs_low)

        return out_high, out_medium, out_low
    
class Stage03Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48,96,kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96)
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(48,48,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48,192,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(192)
        )
        self.medium_to_low=nn.Sequential(
            nn.Conv2d(96,192,kernel_size=3, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(192)
        )
        self.medium_to_high=nn.Sequential(
            nn.Conv2d(96,48,kernel_size=1, bias=False),
            nn.BatchNorm2d(48)
        )
        self.low_to_high=nn.Sequential(
            nn.Conv2d(192, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48)
        )
        self.low_to_medium =nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96)
        )
        self.relu=nn.ReLU()
    def forward(self, inputs_high, inputs_medium, inputs_low):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        medium_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])
        
        low2high=F.interpolate(
            inputs_low, size=high_size, mode="bilinear", align_corners=True
        
        )
        low2high = self.low_to_high(low2high)

        low2med= F.interpolate(
            inputs_low, size=medium_size, mode="bilinear", align_corners=True
        )
        low2med = self.low_to_medium(low2med)

        med2high = F.interpolate(
            inputs_medium, size=high_size, mode="bilinear", align_corners=True
        )
        med2high=self.medium_to_high(med2high)
        med2low=self.medium_to_low(inputs_medium)


        high2med=self.high_to_medium(inputs_high)
        high2low=self.high_to_low(inputs_high)

        out_high=inputs_high+med2high+low2high
        out_medium= inputs_medium+high2med+low2med
        out_low= inputs_low+high2low+med2low

        out_high= self.relu(out_high)
        out_medium= self.relu(out_medium)
        out_low=self.relu(out_low)
        return out_high, out_medium, out_low

class Stage04(nn.Module):
    def __init__(self):
        super().__init__()
        high_res_blocks = [StageBlock(48) for _ in range(3)]
        medium_res_blocks =[StageBlock(96) for _ in range(3)]
        low_res_blocks =[StageBlock(192) for _ in range(3)]
        extra_res_blocks =[StageBlock(384) for _ in range(3)]

        self.high_res_blocks = nn.Sequential(*high_res_blocks)
        self.medium_res_blocks = nn.Sequential(*medium_res_blocks)
        self.low_res_blocks = nn.Sequential(*low_res_blocks)
        self.extra_res_blocks = nn.Sequential(*extra_res_blocks)

    def forward(self, inputs_high, inputs_medium,inputs_low, inputs_extra):
        out_high = self.high_res_blocks(inputs_high)
        out_medium = self.medium_res_blocks(inputs_medium)
        out_low = self.low_res_blocks(inputs_low)
        out_extra = self.extra_res_blocks(inputs_extra)
        return out_high, out_medium, out_low, out_extra
    

class Stage04Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48,96,kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96)
        )
        self.high_to_low = nn.Sequential(
            nn.Conv2d(48,48,kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48,192,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(192)
        )
        self.high_to_extra = nn.Sequential(
            nn.Conv2d(48,48,kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48,48,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48,384,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(384)
        )

        
        self.medium_to_high=nn.Sequential(
            nn.Conv2d(96,48,kernel_size=1, bias=False),
            nn.BatchNorm2d(48)
        )
        self.medium_to_low=nn.Sequential(
            nn.Conv2d(96,192,kernel_size=3, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(192)
        )
        self.medium_to_extra=nn.Sequential(
            nn.Conv2d(96,96,kernel_size=3,stride=2, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Conv2d(96,384,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(384)
        )

        self.low_to_high=nn.Sequential(
            nn.Conv2d(192, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48)
        )
        self.low_to_medium =nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96)
        )
        self.low_to_extra= nn.Sequential(
            nn.Conv2d(192,384,kernel_size=3, stride=2,padding=1, bias=False),
            nn.BatchNorm2d(384)
        )

        self.extra_to_high=nn.Sequential(
            nn.Conv2d(384, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48)
        )
        self.extra_to_medium =nn.Sequential(
            nn.Conv2d(384, 96, kernel_size=1, bias=False),
            nn.BatchNorm2d(96)
        )
        self.extra_to_low =nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192)
        )

        self.relu=nn.ReLU()


    def forward(self, inputs_high, inputs_medium, inputs_low, inputs_extra):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        medium_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])
        low_size = (inputs_low.shape[-1], inputs_low.shape[-2])

        extra2high=F.interpolate(
            inputs_extra, size=high_size, mode="bilinear", align_corners=True
        )
        extra2high = self.extra_to_high(extra2high)

        extra2med= F.interpolate(
            inputs_extra, size=medium_size, mode="bilinear", align_corners=True
        )
        extra2med = self.low_to_medium(extra2med)

        extra2low= F.interpolate(
            inputs_extra, size=low_size, mode="bilinear", align_corners=True
        )
        extra2low = self.low_to_medium(extra2low)



        low2high=F.interpolate(
            inputs_low, size=high_size, mode="bilinear", align_corners=True
        )
        low2high = self.low_to_high(low2high)

        low2med= F.interpolate(
            inputs_low, size=medium_size, mode="bilinear", align_corners=True
        )
        low2med = self.low_to_medium(low2med)

        low2exta=self.low_to_extra(inputs_low)


        med2high = F.interpolate(
            inputs_medium, size=high_size, mode="bilinear", align_corners=True
        )
        med2high=self.medium_to_high(med2high)
        med2low=self.medium_to_low(inputs_medium)
        med2extra=self.medium_to_extra(inputs_medium)

        high2med=self.high_to_medium(inputs_high)
        high2low=self.high_to_low(inputs_high)
        high2extra=self.high_to_extra(inputs_high)

        out_high=inputs_high+med2high+low2high+extra2high
        out_medium= inputs_medium+high2med+low2med+extra2med
        out_low= inputs_low+high2low+med2low+extra2low
        out_extra= inputs_extra+high2extra+med2extra+low2exta

        out_high= self.relu(out_high)
        out_medium= self.relu(out_medium)
        out_low=self.relu(out_low)
        out_extra=self.relu(out_extra)
        return out_high, out_medium, out_low, out_extra
    
class LastBlock(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        total_channels=48+96+192+384
        self.block = nn.Sequential(
            nn.Conv2d(total_channels, total_channels,kernel_size=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.ReLU(),

            nn.Conv2d(total_channels,num_classes, kernel_size=1, bias=False) #or kernel_size=3
        )
    def forward(self, inputs_high, inputs_medium, inputs_low, inputs_vlow):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        original_size=(high_size[0]*4, high_size[1]*4)

        med2high = F.interpolate(inputs_medium, size=high_size, mode="bilinear", align_corners=True)
        low2high = F.interpolate(inputs_low, size=high_size, mode="bilinear", align_corners=True)
        volw2high = F.interpolate(inputs_vlow, size=high_size, mode="bilinear", align_corners=True)
        out = torch.cat([inputs_high,med2high,low2high,volw2high], dim=1)
        out = self.block(out)

        out = F.interpolate(out, size=original_size, mode="bilinear", align_corners=True)
        return out



class HRNet(nn.Module):
    def __init__(self, num_classes):
        super(HRNet, self).__init__()
        # Stage 1
        self.stage01_block = Stage01Block(in_channels=64)
        self.stage01_stream_generate_block = Stage01StreamGenerateBlock()
        
        # Stage 2
        self.stage02 = Stage02()
        self.stage02_fuse = Stage02Fuse()
        self.stage02_stream_geterate_block = StreamGenerateBlock(in_channels=96)
        
        # Stage 3
        self.stage03 = Stage03()
        self.stage03_fuse = Stage03Fuse()
        self.stage03_stream_generate_block = StreamGenerateBlock(in_channels=192)

        # Stage 4
        self.stage04 = Stage04()
        self.stage04_fuse = Stage04Fuse()
        
        # Final Classification Block
        self.last_block = LastBlock(num_classes=num_classes)
    
    def forward(self, x):
        # Stage 1
        x = self.stage01_block(x)
        x_high, x_medium = self.stage01_stream_generate_block(x)
        
        # Stage 2
        x_high, x_medium = self.stage02(x_high, x_medium)
        x_high, x_medium = self.stage02_fuse(x_high, x_medium)
        
        # Stage 3
        x_low = self.stage03_stream_generate_block(x_medium)
        x_high, x_medium, x_low = self.stage03_fuse(x_high, x_medium, x_low)
        
        # Stage 4
        x_vlow = self.stage04_fuse(x_high, x_medium, x_low, x_low)
        x_high, x_medium, x_low, x_vlow = self.stage04(x_high, x_medium, x_low, x_vlow)
        
        # Final Classification Block
        output = self.last_block(x_high, x_medium, x_low, x_vlow)
        
        return output