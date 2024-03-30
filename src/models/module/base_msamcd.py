import torchvision
import math
from torch import Tensor, reshape, stack, mean, max, cat
from typing import Optional, List
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    PReLU,
    ModuleList,
    Sequential,
    ConvTranspose2d,
    Module,
    AdaptiveAvgPool2d,
    Sigmoid,
    Conv1d
)

def getBackbone(
    bkbn_name, pretrained, freeze_backbone
) -> ModuleList:
    # The whole model:
    entire_model = getattr(torchvision.models, bkbn_name)(
        pretrained=pretrained
    ).features
    # Slicing it:
    stage1 = Sequential(
        entire_model[0],
    )
    stage2 = Sequential(
        entire_model[1],
        entire_model[2]
    )
    stage3 = Sequential(
        entire_model[3]
    )
    # used:
    derived_model = ModuleList([
        stage1,
        stage2,
        stage3
    ])

    # Freezing the backbone weights:
    if freeze_backbone:
        for param in derived_model.parameters():
            param.requires_grad = False
    return derived_model


class MCM(Module):
    def __init__(
        self, ch_in: int, ch_out: int, b=1, gamma=2
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            InstanceNorm2d(ch_out),
            PReLU(),
        )

        kernel_size = int(abs((math.log(ch_out, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv = Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        # Mixing:
        mixed = self._convmix(mixed)
        # eca mask
        m = self.avg_pool(mixed)
        m = self.conv(m.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        m = self.sigmoid(m)
        return mixed * m.expand_as(mixed)

class FFUM(Module):
    def __init__(
        self,
        c:int,
        nin: int,
        nout: int,
    ):
        super().__init__()

        self._upsample =Sequential(
            ConvTranspose2d(in_channels=nin, out_channels=nin,
                            kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            InstanceNorm2d(nin),
            PReLU(),
        )

        self._convolution = Sequential(
            Conv2d(c, nin, 3, 1, padding=1),
            InstanceNorm2d(nin),
            PReLU(),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            InstanceNorm2d(nout),
            PReLU(),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)

        if y is not None:
            x = cat((x, y), dim=1)
        return self._convolution(x)

class Head(Module):
    def __init__(
        self,
        fin: List[int],
        fout: List[int],
        last_activation: Module = None,
    ) -> None:
        assert len(fout) == len(fin)
        super().__init__()

        n = len(fin)
        self._linears = Sequential(
            *[
                Sequential(
                    Conv2d(fin[i], fout[i], kernel_size=1, bias=True),
                    PReLU()
                    if i < n - 1 or last_activation is None
                    else last_activation,
                )
                for i in range(n)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        # Processing the tensor:
        return self._linears(x)

class MSAM(Module):
    def __init__(self, in_c, kernel_size=7):
        super(MSAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv7x7 = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = Sigmoid()

        self.conv3x3 = Sequential(
            Conv2d(in_c, in_c//2, 3, 1, padding=1),
            PReLU()
        )
    def forward(self, x):

        t = x
        avg_out = mean(x, dim=1, keepdim=True)
        max_out, _ = max(x, dim=1, keepdim=True)
        out = cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv7x7(out))
        x = out * x

        mix = cat((x, t), dim=1)
        mix = self.conv3x3(mix)
        return mix

