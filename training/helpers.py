from collections import namedtuple
import torch
from torch.nn import Conv2d, BatchNorm2d, PReLU, ReLU, Sigmoid, MaxPool2d, AdaptiveAvgPool2d, Sequential, Module, InstanceNorm2d

import torch.nn.functional as F
"""
ArcFace implementation from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)
"""


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """ A named tuple describing a ResNet block. """


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 50:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=128, num_units=9),
            get_block(in_channel=128, depth=256, num_units=5),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8),
            get_block(in_channel=128, depth=256, num_units=36),
            get_block(in_channel=256, depth=512, num_units=3)
        ]
    else:
        raise ValueError("Invalid number of layers: {}. Must be one of [50, 100, 152]".format(num_layers))
    return blocks


class SEModule(Module):
    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_ORIG(Module):
    expansion = 4

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_ORIG, self).__init__()
        in_planes = in_channel
        planes = depth
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = InstanceNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = InstanceNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = InstanceNorm2d(self.expansion*planes)

        self.shortcut = Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = Sequential(
                Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                InstanceNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.leaky_relu_(self.bn1(self.conv1(x)))
        out = F.leaky_relu_(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.leaky_relu_(out)
        return out


class FPN(Module):
    def __init__(self, block, input_nc, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2d(input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = InstanceNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Top layer
        self.toplayer = Conv2d(2048, 512, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.smooth2 = Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = Conv2d( 512, 512, kernel_size=1, stride=1, padding=0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return Sequential(*layers)

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
        # Bottom-up
        c1 = F.leaky_relu_(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        print(c5.shape)
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        # Smooth
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        return p3, p4, p5

def FPN101(input_nc):
    return FPN(bottleneck_ORIG, input_nc, [2, 2, 2, 2])

class bottleneck_IR(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                BatchNorm2d(depth)
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False), PReLU(depth),
            Conv2d(depth, depth, (3, 3), stride, 1, bias=False), BatchNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(Module):
    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channel, depth, (1, 1), stride, bias=False),
            )
        self.res_layer = Sequential(
            Conv2d(in_channel, depth, kernel_size=1, stride=1, padding=0, bias=False),
            Conv2d(depth, depth, kernel_size=3, stride=1, padding=1, bias=False, groups=depth),
            PReLU(depth),
            InstanceNorm2d(depth),
            Conv2d(depth, depth, kernel_size=1, stride=1, padding=0, bias=False),
            Conv2d(depth, depth, kernel_size=3, stride=stride, padding=1, bias=False, groups=depth),
            SEModule(depth, 16),
            PReLU(depth),
            InstanceNorm2d(depth)
        )

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut