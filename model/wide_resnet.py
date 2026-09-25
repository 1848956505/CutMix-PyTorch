import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, drop_rate=0.0):
        super().__init__()

        self.equal_channels = in_channels == out_channels

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.drop_rate = drop_rate

        if self.equal_channels:
            self.shortcut = None
        else:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False
            )

    def forward(self, x):

        if self.equal_channels:
            out = self.relu1(self.bn1(x))
            shortcut = x
        else:
            x_act = self.relu1(self.bn1(x))
            out = x_act
            shortcut = self.shortcut(x_act)

        out = self.conv1(out)
        out = self.relu2(self.bn2(out))

        if self.drop_rate > 0:
            out = F.dropout(
                out,
                p=self.drop_rate,
                training=self.training
            )

        out = self.conv2(out)

        return shortcut + out


class NetworkBlock(nn.Module):

    def __init__(
        self,
        num_layers,
        in_channels,
        out_channels,
        stride,
        drop_rate
    ):
        super().__init__()

        layers = []

        for i in range(num_layers):

            layers.append(
                BasicBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    drop_rate
                )
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class WideResNet(nn.Module):

    def __init__(
        self,
        depth=28,
        widen_factor=10,
        num_classes=10,
        drop_rate=0.0 # Mixup论文没有使用dropout
    ):
        super().__init__()

        assert (depth - 4) % 6 == 0

        blocks_per_group = (depth - 4) // 6

        channels = [
            16,
            16 * widen_factor,
            32 * widen_factor,
            64 * widen_factor
        ]

        self.conv1 = nn.Conv2d(
            3,
            channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.block1 = NetworkBlock(
            blocks_per_group,
            channels[0],
            channels[1],
            stride=1,
            drop_rate=drop_rate
        )

        self.block2 = NetworkBlock(
            blocks_per_group,
            channels[1],
            channels[2],
            stride=2,
            drop_rate=drop_rate
        )

        self.block3 = NetworkBlock(
            blocks_per_group,
            channels[2],
            channels[3],
            stride=2,
            drop_rate=drop_rate
        )

        self.bn = nn.BatchNorm2d(channels[3])
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Linear(
            channels[3],
            num_classes
        )

        self.final_channels = channels[3]

        self._initialize_weights()

    def _initialize_weights(self):

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.conv1(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.relu(self.bn(x))

        x = F.avg_pool2d(x, 8)

        x = x.view(
            x.size(0),
            self.final_channels
        )

        return self.fc(x)


def wide_resnet28_10(num_classes=10):

    return WideResNet(
        depth=28,
        widen_factor=10,
        num_classes=num_classes,
        drop_rate=0.0
    )