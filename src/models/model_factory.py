import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50, x3d_s

def get_model(name="slowfast", num_classes=101):
    if name == "slowfast":
        model = slowfast_r50(pretrained=True)
    elif name == "x3d":
        model = x3d_s(pretrained=True)
    else:
        raise ValueError(f"Unknown model name: {name}")

    model.blocks[-1].proj = nn.Linear(model.blocks[-1].proj.in_features, num_classes)
    return model
