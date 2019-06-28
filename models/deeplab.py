import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import ASPP
from models.decoder import Decoder
from models.backbone import build_backbone
import numpy as np


class DeepLab(nn.Module):

    def __init__(self, backbone='mobilenet', output_stride=16, num_classes=19, sync_bn=True, freeze_bn=False, mc_dropout=False, input_channels=3, pretrained=True):

        super(DeepLab, self).__init__()

        if sync_bn == True:
            batchnorm = SynchronizedBatchNorm2d
        else:
            batchnorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, batchnorm, mc_dropout, input_channels, pretrained)
        self.aspp = ASPP(backbone, output_stride, batchnorm)
        self.decoder = Decoder(num_classes, backbone, batchnorm, mc_dropout)
        self.return_features = False
        self.noisy_features = False
        self.model_name = 'deeplab'
        if freeze_bn:
            self.freeze_bn()

    def set_return_features(self, return_features):
        self.return_features = return_features

    def set_noisy_features(self, noisy_features):
        self.noisy_features = noisy_features

    def forward(self, input):

        if self.noisy_features is True:
            noise_input = np.random.normal(loc=0.0, scale=abs(input.mean().cpu().item() * 0.05), size=input.shape).astype(np.float32)
            input = input + torch.from_numpy(noise_input).cuda()

        x, low_level_feat = self.backbone(input)

        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            noise_low_level_feat = np.random.normal(loc=0.0, scale=abs(low_level_feat.mean().cpu().item() *
                                                                       0.5), size=low_level_feat.shape).astype(np.float32)
            x += torch.from_numpy(noise_x).cuda()
            low_level_feat += torch.from_numpy(noise_low_level_feat).cuda()

        x = self.aspp(x)

        if self.noisy_features is True:
            noise_x = np.random.normal(loc=0.0, scale=abs(x.mean().cpu().item() * 0.5), size=x.shape).astype(np.float32)
            x += torch.from_numpy(noise_x).cuda()

        low_res_x, features = self.decoder(x, low_level_feat)
        x = F.interpolate(low_res_x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if self.return_features:
            return x, features
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def test_gradients():
    from utils.loss import SegmentationLosses
    import numpy as np
    from dataloaders import make_dataloader
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    from dataloaders.utils import map_segmentation_to_colors
    import sys

    kwargs = {'pin_memory': True, 'init_set': 'set_dummy.txt'}
    _, train_loader, _, _, num_classes = make_dataloader('active_cityscapes_region', 513, 513, 1, True, **kwargs)

    model = DeepLab(backbone='mobilenet', output_stride=16, mc_dropout=False)
    train_params = [{'params': model.get_1x_lr_params(), 'lr': 0.001},
                    {'params': model.get_10x_lr_params(), 'lr': 0.001 * 10}]
    optimizer = torch.optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=False)
    criterion = SegmentationLosses(weight=None, cuda=True).build_loss(mode='ce')
    model = model.cuda()
    model.eval()

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            print(param)
            break

    display = False
    for i, sample in enumerate(train_loader):
        image, target = sample['image'], sample['label']
        image, target = image.cuda(), target.cuda()

        if display:
            gt_colored = map_segmentation_to_colors(np.array(target[0].cpu().numpy()).astype(np.uint8), 'cityscapes')
            image_unnormalized = ((np.transpose(image[0].cpu().numpy(), axes=[1, 2, 0]) *
                                   (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(image_unnormalized)
            plt.subplot(212)
            plt.imshow(gt_colored)
            plt.show()

        optimizer.zero_grad()
        output = model(image)
        loss = criterion(output, target)
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
                print(param.grad)
                input()
        optimizer.step()

    sys.exit(0)

if __name__ == "__main__":

    model = DeepLab(backbone='mobilenet', output_stride=16, mc_dropout=True)
    model.eval()

    def turn_on_dropout(m):
        if type(m) == nn.Dropout2d:
            m.train()

    model.apply(turn_on_dropout)

    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())
    print('NumElements: ', sum([p.numel() for p in model.parameters()]))

    model = DeepLab(backbone='mobilenet', output_stride=16, mc_dropout=True)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    model.set_return_features(True)
    output, features = model(input)
    print(output.size(), F.avg_pool2d(features, (64, 64), 64 // 2).size())
