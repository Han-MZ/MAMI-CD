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
    Conv1d, AdaptiveMaxPool2d, ReLU, BatchNorm2d,
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


class MECAM(Module):
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

class MCAM(Module):
    def __init__(self, ch_in: int, ch_out: int, ratio=16):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            InstanceNorm2d(ch_out),
            PReLU(),
        )

        self.avg_pool = AdaptiveAvgPool2d(1)
        self.max_pool = AdaptiveMaxPool2d(1)
        self.fc = Sequential(
            Conv2d(ch_out, ch_out // ratio, 1, bias=False),
            ReLU(inplace=True),
            Conv2d(ch_out // ratio, ch_out, 1, bias=False)
        )

        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        # Packing the tensors and interleaving the channels:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        # Mixing:
        mixed = self._convmix(mixed)
        # cam mask
        avg_out = self.fc(self.avg_pool(mixed))
        max_out = self.fc(self.max_pool(mixed))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * mixed


class MaskFusionUpsamplingStrategy(Module):
    def __init__(
        self,
        nin: int,
        nout: int,
    ):
        super().__init__()
        # self._upsample = Upsample(
        #     scale_factor=2, mode="bilinear", align_corners=True
        # )
        self._upsample =Sequential(
            ConvTranspose2d(in_channels=nin, out_channels=nin,
                            kernel_size=4, stride=2, padding=1, output_padding=0, bias=False),
            InstanceNorm2d(nin),
            PReLU(),
        )

        self._convolution = Sequential(
            Conv2d(nin, nin, 3, 1, groups=nin, padding=1),
            InstanceNorm2d(nin),
            PReLU(),
            Conv2d(nin, nout, kernel_size=1, stride=1),
            InstanceNorm2d(nout),
            PReLU(),
        )

    def forward(self, x: Tensor, y: Optional[Tensor] = None) -> Tensor:
        x = self._upsample(x)
        if y is not None:
            x = x * y
        return self._convolution(x)


class PixelwiseLinear(Module):
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


class SDAM(Module):
    def __init__(self, in_c, c1, c2, kernel_size=7):
        super(SDAM, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv7x7 = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = Sigmoid()

        self.conv1x1 = Sequential(
            Conv2d(in_c // 2, c1, 1),
            PReLU(),
            Conv2d(c1, c2, 1),
            PReLU(),
            Conv2d(c2, 1, 1),
            Sigmoid(),
        )
        self.conv3x3 = Sequential(
            Conv2d(in_c, in_c // 2, 3, groups=in_c//2, padding=1),
            PReLU()
        )

    def forward(self, x):

        t = x

        avg_out = mean(x, dim=1, keepdim=True)
        max_out, _ = max(x, dim=1, keepdim=True)
        out = cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv7x7(out))
        x = out * x

        mix = stack((x, t), dim=2)
        mix = reshape(mix, (mix.shape[0], -1, mix.shape[3], mix.shape[4]))
        mix = self.conv3x3(mix)
        mix = self.conv1x1(mix)
        return mix

class MixingSdam(Module):
    def __init__(self):
        super().__init__()
        self.mixconv = Conv2d(2,1,1)
        self.sig = Sigmoid()

    def forward(self, m1, m2):
        m = cat([m1, m2], dim=1)
        m = self.sig(self.mixconv(m))
        return m


class EFEM(Module):
    def __init__(self, in_ch, out_ch = 21):
        super().__init__()
        self.conv1x1_ = Sequential(
            Conv2d(in_ch, out_ch, 1, padding=0),
            BatchNorm2d(out_ch),
            ReLU(inplace=True),
        )

        self.dilated_conv3x3 = Sequential(
            Conv2d(out_ch, 10, kernel_size=3, padding=2, dilation=2),
            BatchNorm2d(10),
            ReLU(inplace=True),
        )
        self.score_dsn = Conv2d(21, 1, 1)

        self.attn = CBAM(21)
        self.conv_channel = Conv2d(21, 11, kernel_size=1)
        self.sigmoid = Sigmoid()
    def forward(self, x):
        x = self.conv1x1_(x)
        dil_x = self.dilated_conv3x3(x)
        """add_attn"""
        attn_x = self.attn(x)
        attn_x_1 = self.conv_channel(attn_x)

        muli_x = cat([dil_x, attn_x_1], dim=1)

        score_dsn = self.score_dsn(muli_x)

        edge_feature = self.sigmoid(score_dsn)

        return edge_feature

class ChannelAttention(Module):
    """
    CBAM混合注意力机制的通道注意力
    """

    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.max_pool = AdaptiveMaxPool2d(1)

        self.fc = Sequential(

            # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
            Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            ReLU(inplace=True),
            Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )

        self.sigmoid = Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x

class SpatialAttention(Module):
    """
    CBAM混合注意力机制的空间注意力
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        avg_out = mean(x, dim=1, keepdim=True)
        max_out, _ = max(x, dim=1, keepdim=True)
        out = cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out * x

class CBAM(Module):
    """
    CBAM混合注意力机制
    """

    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = self.channelattention(x)
        x = self.spatialattention(x)
        return x
