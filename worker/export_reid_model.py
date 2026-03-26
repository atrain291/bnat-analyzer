"""Export MobileNetV3-Small as a 512-dim Re-ID feature extractor to ONNX.

Run at Docker build time to bake the model into the image.
"""
import os
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class ReIDNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(576, 512)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    output_path = os.environ.get("REID_MODEL_PATH", "/app/trt_cache/osnet_x025.onnx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = ReIDNet()
    model.eval()
    dummy = torch.randn(1, 3, 256, 128)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["input"],
        output_names=["embedding"],
        opset_version=11,
    )
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Exported Re-ID model to {output_path} ({size_mb:.1f}MB)")
