import torch
from torch import nn
from models.deeplab import DeepLab
from models.unet import UNet
from models.enet import ENet


class DeepLabAccuracyPredictor(nn.Module):

    def __init__(self, backbone, output_stride, num_classes, sync_bn, freeze_bn, mc_dropout, enet=False):
        super(DeepLabAccuracyPredictor, self).__init__()
        if not enet:
            self.deeplab = DeepLab(num_classes=num_classes, backbone=backbone, output_stride=output_stride,
                                   sync_bn=sync_bn, freeze_bn=freeze_bn, mc_dropout=mc_dropout)
        else:
            self.deeplab = ENet(num_classes=num_classes, encoder_relu=True, decoder_relu=True)
        self.unet = UNet(3 + num_classes, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        deeplab_activations = self.deeplab(x)
        unet_input = torch.cat([self.softmax(deeplab_activations.detach()), x], dim=1)
        unet_activations = self.unet(unet_input)
        return deeplab_activations, unet_activations

    def get_1x_lr_params(self):
        return self.deeplab.get_1x_lr_params()

    def get_10x_lr_params(self):
        return self.deeplab.get_10x_lr_params()

    def get_enet_params(self):
        return self.deeplab.parameters()

    def get_unet_params(self):
        return self.unet.parameters()


if __name__ == "__main__":
    from dataloaders.dataset import active_cityscapes
    import os
    import constants
    import matplotlib.pyplot as plt
    from models.sync_batchnorm.replicate import patch_replication_callback
    from torch.utils.data import DataLoader
    from dataloaders.utils import map_segmentations_to_colors, map_segmentation_to_colors, map_binary_output_mask_to_colors
    import numpy as np

    class dotdict(dict):
        """dot.notation access to dictionary attributes"""
        __getattr__ = dict.get
        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

    args = {
        'base_size': 513,
        'crop_size': 513,
        'seed_set': 'set_0.txt',
        'batch_size': 2
    }

    args = dotdict(args)
    dataset_path = os.path.join(constants.DATASET_ROOT, 'cityscapes')
    train_set = active_cityscapes.ActiveCityscapesImage(path=dataset_path, base_size=args.base_size,
                                                        crop_size=args.crop_size, split='train', init_set=args.seed_set, overfit=False)

    model = DeepLabAccuracyPredictor(backbone='mobilenet', output_stride=16, num_classes=19, sync_bn=True, freeze_bn=False, mc_dropout=False)
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    checkpoint = torch.load(os.path.join(constants.RUNS, 'active_cityscapes_image', 'accuracy_predictor_50_point_fit', 'run_0002', 'checkpoint.pth.tar'))
    model.module.load_state_dict(checkpoint['state_dict'])
    model.eval()

    dataloader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    for i, sample in enumerate(dataloader):
        image_batch = sample['image'].cuda()
        dl_out, un_out = model(image_batch)
        un_target = dl_out.argmax(1).cpu().squeeze() == sample['label'].long()
        un_target[sample['label'] == 255] = 255
        for j in range(sample['image'].size()[0]):
            image = sample['image'].cpu().numpy()
            gt = sample['label'].cpu().numpy()
            gt_colored = map_segmentation_to_colors(np.array(gt[j]).astype(np.uint8), 'cityscapes')
            image_unnormalized = ((np.transpose(image[j], axes=[1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)
            plt.figure()
            plt.title('plot')
            plt.subplot(231)
            plt.imshow(image_unnormalized)
            plt.subplot(232)
            plt.imshow(map_segmentation_to_colors(np.array(un_target.numpy()[j]).astype(np.uint8), 'binary'))
            plt.subplot(233)
            plt.imshow(gt_colored)
            plt.subplot(236)
            plt.imshow(map_segmentation_to_colors(np.array(dl_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'cityscapes'))
            plt.subplot(235)
            plt.imshow(map_segmentation_to_colors(np.array(un_out.argmax(1).detach().cpu().numpy()[j]).astype(np.uint8), 'binary'))
            plt.show(block=True)

    deeplab_activations, unet_activations = model(input)
