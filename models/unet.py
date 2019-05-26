import torch
import torch.nn.functional as F
from torch import nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )


class UNet(nn.Module):

    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()

        self.dconv_down1 = double_conv(in_channels, 32)
        self.dconv_down2 = double_conv(32, 64)
        self.dconv_down3 = double_conv(64, 128)
        self.dconv_down4 = double_conv(128, 256)
        self.dropout = nn.Dropout2d()

        self.maxpool = nn.MaxPool2d(2)

        self.dconv_up3 = double_conv(128 + 256, 128)
        self.dconv_up2 = double_conv(64 + 128, 64)
        self.dconv_up1 = double_conv(64 + 32, 32)

        self.conv_last = nn.Conv2d(32, num_classes, 1)
        self._initialize_weights()

    def forward(self, x):
        input_size = x.size()[2:]
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = self.dconv_down4(x)
        x = F.interpolate(x, conv3.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        x = F.interpolate(x, conv2.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        x = F.interpolate(x, conv1.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        out = self.conv_last(x)
        return F.interpolate(out, input_size, mode='bilinear', align_corners=True)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


if __name__ == '__main__':

    model = UNet(3, 2)
    model.cuda()
    model.eval()
    input = torch.cuda.FloatTensor(1, 3, 513, 513)

    print('NumElements: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    out = model(input)
    print(out.shape)
