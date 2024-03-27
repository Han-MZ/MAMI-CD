
from torch import Tensor
from src.utils.grad_cam.model.excd_base import (
    SDAM,
    getBackbone,
    MECAM,
    MaskFusionUpsamplingStrategy,
    PixelwiseLinear,
)
from torch.nn import ModuleList, Module, Sigmoid

class MECACD(Module):
    def __init__(self, bkbn_name:str):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self._backbone = getBackbone(bkbn_name=bkbn_name, pretrained=True, freeze_backbone=False)
        # Initialize sdam blocks:
        self._mixing_mask = ModuleList(
            [
                SDAM(6, 6, 3),
                SDAM(48, 12, 6),
                SDAM(96, 24, 12)
            ]
        )
        # Initialize Upsampling blocks:
        self._up = ModuleList([
            MaskFusionUpsamplingStrategy(80, 64),
            MaskFusionUpsamplingStrategy(64, 64),
            MaskFusionUpsamplingStrategy(64, 32)
        ])
        # Initialize exchange blocks:
        self._ex = MECAM(160, 80)
        # Initialize mixing blocks:
        self._mecam = MECAM(64, 32)
        # Final classification layer:
        self._classify = PixelwiseLinear([32, 16, 8], [16, 8, 1], Sigmoid())

    def forward(self, ref: Tensor, test: Tensor) -> Tensor:
        features_1, features_2 = self._encode(ref, test)
        latents_1, latents_2 = self._decode(features_1, features_2)
        mixd = self._mecam(latents_1, latents_2)

        return self._classify(mixd)

    def _encode(self, ref, test):
        features_1 = []
        features_2 = []
        for num, layer in enumerate(self._backbone):
            features_1.append(self._mixing_mask[num](ref))
            features_2.append(self._mixing_mask[num](test))
            ref, test = layer(ref), layer(test)

        mix = self._ex(ref, test)
        features_1.append(mix)
        features_2.append(mix)

        return features_1,features_2

    def _decode(self, features_1, features_2) -> Tensor:
        upping_1 = features_1[-1]
        upping_2 = features_2[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping_1 = self._up[i](upping_1, features_1[j])
            upping_2 = self._up[i](upping_2, features_2[j])
        return upping_1, upping_2
