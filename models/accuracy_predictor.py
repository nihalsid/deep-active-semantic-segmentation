import torch
from torch import nn
from models.deeplab import DeepLab
from models.unet import UNet


class DeepLabAccuracyPredictor(nn.Module):

    def __init__(self, backbone, output_stride, num_classes, sync_bn, freeze_bn, mc_dropout):
        super(DeepLabAccuracyPredictor, self).__init__()
        self.deeplab = DeepLab(num_classes=num_classes, backbone=backbone, output_stride=output_stride,
                               sync_bn=sync_bn, freeze_bn=freeze_bn, mc_dropout=mc_dropout)
        self.unet = UNet(3 + num_classes, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        deeplab_activations = self.deeplab(x)
        unet_input = torch.cat([self.softmax(deeplab_activations), x], dim=1)
        unet_activations = self.unet(unet_input)
        return deeplab_activations, unet_activations

    def get_1x_lr_params(self):
        return self.deeplab.get_1x_lr_params()

    def get_10x_lr_params(self):
        return self.deeplab.get_10x_lr_params()

    def get_unet_params(self):
        return self.unet.parameters()


if __name__ == "__main__":

    model = DeepLabAccuracyPredictor(backbone='mobilenet', output_stride=16, num_classes=19, sync_bn=True, freeze_bn=False, mc_dropout=False)
    model.eval()

    input = torch.rand(1, 3, 512, 512)
    deeplab_activations, unet_activations = model(input)
    print(deeplab_activations.size(), unet_activations.size())
    print('NumElements: ', sum([p.numel() for p in model.parameters()]))
