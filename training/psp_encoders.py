import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, InstanceNorm2d, PReLU, Sequential, Module, LeakyReLU, Embedding, MaxPool2d, InstanceNorm1d

from .helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE, FPN101
from .networks import FullyConnectedLayer


class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c//4, kernel_size=1, stride=1, padding=0),
                    Conv2d(out_c//4, out_c//4, kernel_size=3, stride=2, padding=1, groups=out_c//4),
                    nn.LeakyReLU(inplace=True),
                    Conv2d(out_c // 4, out_c, kernel_size=1, stride=1, padding=0),
                    nn.LeakyReLU(inplace=True),
                    nn.InstanceNorm2d(out_c),
                    ]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c//4, kernel_size=1, stride=1, padding=0),
                Conv2d(out_c//4, out_c//4, kernel_size=3, stride=2, padding=1, groups=out_c//4),
                nn.LeakyReLU(inplace=True),
                Conv2d(out_c // 4, out_c, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=True),
                nn.InstanceNorm2d(out_c)
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = FullyConnectedLayer(out_c, out_c)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x



class GradualStyleEncoder2(Module):
    def __init__(self, input_nc=3):
        super(GradualStyleEncoder2, self).__init__()

        self.fpn = FPN101(input_nc)
        self.embed = Embedding(18, 512)

        self.pool = MaxPool2d(2, 2)

        self.map2style1 = Sequential(
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            LeakyReLU(inplace=True),
            InstanceNorm2d(512),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            LeakyReLU(inplace=True),
            InstanceNorm2d(512),
        )

        self.map2style2 = Sequential(
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            LeakyReLU(inplace=True),
            InstanceNorm2d(512),
        )

        self.map2style3 = Sequential(
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            LeakyReLU(inplace=True),
            InstanceNorm2d(512),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            LeakyReLU(inplace=True),
            InstanceNorm2d(512),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            LeakyReLU(inplace=True),
            InstanceNorm2d(512),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            LeakyReLU(inplace=True),
            InstanceNorm2d(512),
            Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            LeakyReLU(inplace=True),
            InstanceNorm2d(512),
        )

        self.last1 = Sequential(
            FullyConnectedLayer(1024, 512),
            LeakyReLU(inplace=True),
        )
        self.last2 = Sequential(
            FullyConnectedLayer(1024, 512),
            LeakyReLU(inplace=True),
            FullyConnectedLayer(512, 512),
            LeakyReLU(inplace=True),
            FullyConnectedLayer(512, 512)
        )

    def forward(self, x):
        p3, p4, p5 = self.fpn(x)
        print(p3.shape, p4.shape, p5.shape)
        p3 = self.map2style1(p3)
        p3 = self.map2style2(p3)
        p3 = self.map2style3(p3)
        p3 = p3.view(-1, 512)

        p4 = self.map2style2(p4)
        p4 = self.map2style3(p4)
        p4 = p4.view((-1, 512))
        p4 = torch.cat((p3, p4), dim=1).view(-1, 1024)
        p4 = self.last1(p4)

        p5 = self.map2style3(p5)
        p5 = p5.view(-1, 512)
        p5 = torch.cat((p4, p5), dim=1).view(-1, 1024)
        p5 = self.last1(p5)

        print(p3.shape, p4.shape, p5.shape)
        indices = self.embed(torch.arange(0, 18, device=x.device))[None, ...]
        print(indices.shape)
        p = torch.repeat_interleave(torch.stack((p3, p4, p5), dim=1).unsqueeze(2), 6, dim=2).flatten(1, 2)
        print(p.shape)
        p = torch.cat((p, indices))
        print(p.shape)
        out = self.last2(p)
        out = self.linear(out)
        return out


class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', input_nc=3):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(input_nc, 64, (3, 3), 1, 1, bias=False),
                                      InstanceNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = 18
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 32)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 64)
            else:
                style = GradualStyleBlock(512, 512, 128)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 15:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x
        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))
        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))
        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))
        out = torch.stack(latents, dim=1)
        return out


class BackboneEncoderUsingLastLayerIntoW(Module):
    def __init__(self, num_layers, mode='ir', input_nc=3):
        super(BackboneEncoderUsingLastLayerIntoW, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoW')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(input_nc, 64, (3, 3), 1, 1, bias=False),
                                      InstanceNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = FullyConnectedLayer(512, 512)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_pool(x)
        x = x.view(-1, 512)
        x = self.linear(x)
        return x


class BackboneEncoderUsingLastLayerIntoWPlus(Module):
    def __init__(self, num_layers, mode='ir', input_nc=3):
        super(BackboneEncoderUsingLastLayerIntoWPlus, self).__init__()
        print('Using BackboneEncoderUsingLastLayerIntoWPlus')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(input_nc, 64, (3, 3), 1, 1, bias=False),
                                      InstanceNorm2d(64),
                                      PReLU(64))
        self.output_layer_2 = Sequential(InstanceNorm2d(512),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(512 * 7 * 7, 512))
        self.linear = FullyConnectedLayer(512, 512 * 18)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer_2(x)
        x = self.linear(x)
        x = x.view(-1, 18, 512)
        return x