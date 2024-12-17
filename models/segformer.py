import torch
from transformers import SegformerForSemanticSegmentation

class SegFormer_B0(torch.nn.Module):
    def __init__(self, num_classes=29):
        super(SegFormer_B0, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512", num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, image):
        outputs = self.model(pixel_values=image)
        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits, size=image.shape[-1], mode="bilinear", align_corners=False)
        return upsampled_logits
