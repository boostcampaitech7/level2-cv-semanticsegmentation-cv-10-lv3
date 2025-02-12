from transformers import UperNetForSemanticSegmentation

class UperNet_ConvNext_xlarge(nn.Module):
    def __init__(self, num_classes=29):
        super(UperNet_ConvNext_xlarge, self).__init__()
        self.model = UperNetForSemanticSegmentation.from_pretrained(
            "openmmlab/upernet-convnext-xlarge", num_labels=num_classes, ignore_mismatched_sizes=True
        )

    def forward(self, image):
        outputs = self.model(pixel_values=image)
        return outputs.logits