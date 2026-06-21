import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic block: Conv -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_c, out_c, k, s, p, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(out_c, eps=1e-5) if bn else None
        self.act = nn.LeakyReLU(0.1, inplace=True) if bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn:
            x = self.act(self.bn(x))
        return x


class ResBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels // 2, 1, 1, 0)
        self.conv2 = ConvBlock(channels // 2, channels, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class YOLOv3(nn.Module):
    """Complete YOLOv3 architecture with Darknet-53 backbone"""
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes

        # Backbone: Darknet-53
        self.conv1 = ConvBlock(3, 32, 3, 1, 1)
        self.layer1 = self._make_layer(32, 64, 1)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 8)
        self.layer4 = self._make_layer(256, 512, 8)
        self.layer5 = self._make_layer(512, 1024, 4)

        # Head 1: 13x13 scale (large objects)
        self.head1_1 = self._make_c5(1024, 512)
        self.head1_2 = self._make_yolo_head(512, 1024, num_classes)

        # Head 2: 26x26 scale (medium objects)
        self.head2_1 = ConvBlock(512, 256, 1, 1, 0)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.head2_2 = self._make_c5(768, 256)
        self.head2_3 = self._make_yolo_head(256, 512, num_classes)

        # Head 3: 52x52 scale (small objects)
        self.head3_1 = ConvBlock(256, 128, 1, 1, 0)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.head3_2 = self._make_c5(384, 128)
        self.head3_3 = self._make_yolo_head(128, 256, num_classes)

    def _make_layer(self, in_c, out_c, num_blocks):
        layers = [ConvBlock(in_c, out_c, 3, 2, 1)]
        for _ in range(num_blocks):
            layers.append(ResBlock(out_c))
        return nn.Sequential(*layers)

    def _make_c5(self, in_c, out_c):
        """5 interleaved convolution blocks"""
        return nn.Sequential(
            ConvBlock(in_c, out_c, 1, 1, 0),
            ConvBlock(out_c, out_c * 2, 3, 1, 1),
            ConvBlock(out_c * 2, out_c, 1, 1, 0),
            ConvBlock(out_c, out_c * 2, 3, 1, 1),
            ConvBlock(out_c * 2, out_c, 1, 1, 0),
        )

    def _make_yolo_head(self, in_c, out_c, num_classes):
        """Final linear convolution without BatchNorm for box prediction"""
        return nn.Sequential(
            ConvBlock(in_c, out_c, 3, 1, 1),
            nn.Conv2d(out_c, 3 * (5 + num_classes), 1, 1, 0, bias=True)
        )
def forward(self, x):
    x = self.layer2(self.layer1(self.conv1(x)))
    route1 = self.layer3(x)
    route2 = self.layer4(route1)
    x = self.layer5(route2)

    x1_5 = self.head1_1(x)
    out1 = self.head1_2(x1_5)

    x = self.upsample1(self.head2_1(x1_5))
    x = torch.cat([x, route2], dim=1)
    x2_5 = self.head2_2(x)
    out2 = self.head2_3(x2_5)

    x = self.upsample2(self.head3_1(x2_5))
    x = torch.cat([x, route1], dim=1)
    x3_5 = self.head3_2(x)
    out3 = self.head3_3(x3_5)

    return out1, out2, out3
 
