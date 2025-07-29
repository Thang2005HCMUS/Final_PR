import torch
import logging
import clip

from transformers import Dinov2Model, Dinov2Config
from .dinov2.models.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

class ClipModel():
    def __init__(self, rank, backbone_size):
        self.backbone, _ = clip.load(backbone_size, device="cuda", jit=False)
        self.backbone.to(rank)

        for param in self.backbone.parameters():
            if param.dtype == torch.float16:
                param.data = param.data.to(torch.float32)

class DINOv2Model():
    def __init__(self, rank, backbone_size):
        self.backbone = Dinov2Model.from_pretrained("facebook/dinov2-" + backbone_size)
        self.backbone.to(rank)



class Dinov2BaselineModel():
    def __init__(self, rank, backbone_size, patch_size=14, init_values=0.1):
        logging.info("Loading scratch vit " + backbone_size + " ...")

        backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }

        self.backbone_arch = backbone_archs[backbone_size]
        self.backbone_name = f"dinov2_{self.backbone_arch}"

        if backbone_size == "small":
            self.backbone = vit_small(patch_size=patch_size, init_values=init_values, block_chunks=False)
        elif backbone_size == "base":
            self.backbone = vit_base(patch_size=patch_size, init_values=init_values, block_chunks=False)
        elif backbone_size == "large":
            self.backbone = vit_large(patch_size=patch_size, init_values=init_values, block_chunks=False)
        elif backbone_size == "giant":
            self.backbone = vit_giant2(patch_size=patch_size, init_values=init_values, block_chunks=False)
        self.backbone.to(rank)
