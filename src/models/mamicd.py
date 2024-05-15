
from torch import Tensor
from models.module.base_mamicd import (
    MAM,
    getBackbone,
    MIM,
    FFUM,
    Head,
)
from torch.nn import ModuleList, Module, Sigmoid

class MAMICD(Module):
    def __init__(self, bkbn_name:str):
        super().__init__()

        # Load the pretrained backbone according to parameters:
        self._backbone = getBackbone(bkbn_name=bkbn_name, pretrained=True, freeze_backbone=False)
        # Initialize sdam blocks:
        self._mixing_mask = ModuleList(
            [
                MAM(6),
                MAM(48),
                MAM(96)
            ]
        )
        # Initialize Upsampling blocks:
        self._up = ModuleList([
            FFUM(128, 80, 64),
            FFUM(88, 64, 64),
            FFUM(67, 64, 32)
        ])
        # Initialize exchange blocks:
        self._ex = MIM(160, 80)
        # Initialize mixing blocks:
        self._mecam = MIM(64, 32)
        # Final classification layer:
        self._classify = Head([32, 16, 8], [16, 8, 1], Sigmoid())

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

    def _decode(self, features_1, features_2) :
        upping_1 = features_1[-1]
        upping_2 = features_2[-1]
        for i, j in enumerate(range(-2, -5, -1)):
            upping_1 = self._up[i](upping_1, features_1[j])
            upping_2 = self._up[i](upping_2, features_2[j])
        return upping_1, upping_2
