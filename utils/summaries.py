import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import map_segmentations_to_colors, map_segmentation_to_colors, map_binary_output_mask_to_colors
import scipy.misc
import constants
import numpy as np


class TensorboardSummary:

    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=self.directory)
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step, prefix='val'):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image(f'{prefix}/Image', grid_image, global_step)
        grid_image = make_grid(map_segmentations_to_colors(torch.max(output[:3], 1)[
                               1].detach().cpu().numpy(), dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image(f'{prefix}/Prediction', grid_image, global_step)
        grid_image = make_grid(map_segmentations_to_colors(torch.squeeze(
            target[:3], 1).detach().cpu().numpy(), dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image(f'{prefix}/Groundtruth', grid_image, global_step)

    def visualize_image_with_unet(self, writer, dataset, image, target_0, output_0, target_1, output_1, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image('Image', grid_image, global_step)
        grid_image = make_grid(map_segmentations_to_colors(torch.max(output_0[:3], 1)[
                               1].detach().cpu().numpy(), dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted label', grid_image, global_step)
        grid_image = make_grid(map_segmentations_to_colors(torch.squeeze(
            target_0[:3], 1).detach().cpu().numpy(), dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth label', grid_image, global_step)
        grid_image = make_grid(map_binary_output_mask_to_colors(torch.max(output_1[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
        writer.add_image('Predicted Accuracy', grid_image, global_step)
        grid_image = make_grid(map_binary_output_mask_to_colors(torch.squeeze(
            target_1[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
        writer.add_image('Groundtruth Accuracy', grid_image, global_step)

    def create_single_visualization(self, writer, name, dataset, image, target_0, output_0, target_1, output_1, global_step):
        for k in range(min(3, image.shape[0])):
            tensor_image = torch.from_numpy(np.transpose((np.transpose(image[k].clone().cpu().numpy(), axes=[
                                            1, 2, 0]) * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)), [2, 0, 1])).float().unsqueeze(0)
            target_0_image = map_segmentations_to_colors(torch.squeeze(target_0[k:k + 1], 1).detach().cpu().numpy(), dataset=dataset).float()
            output_0_image = map_segmentations_to_colors(torch.max(output_0[k:k + 1], 1)[1].detach().cpu().numpy(), dataset=dataset).float()
            target_1_image = map_binary_output_mask_to_colors(torch.squeeze(target_1[k:k + 1], 1).detach().cpu().numpy()).float()
            output_1_image = map_binary_output_mask_to_colors(torch.max(output_1[k:k + 1], 1)[1].detach().cpu().numpy()).float()
            zeros_image = torch.zeros_like(target_0_image).cpu().float()
            torch.cat((tensor_image, output_0_image, output_1_image), -1)
            torch.cat((zeros_image, target_0_image, target_1_image), -1)
            grid_image = make_grid(torch.cat((torch.cat((tensor_image, output_0_image, output_1_image), -1),
                                              torch.cat((zeros_image, target_0_image, target_1_image), -1)), -2), 3, normalize=False, range=(0, 255))
            writer.add_image('%s/accuracy_predictions_%d' % (name, k), grid_image, global_step)

    @staticmethod
    def visualize_images_to_folder(output_folder, batch_idx, image, target, output, dataset):

        for i in range(image.shape[0]):
            outpath = os.path.join(output_folder, '{0}_{1:04d}_{2:02d}.png')
            scipy.misc.imsave(outpath.format('img', batch_idx, i), np.transpose(image[i, :, :, :], axes=[1, 2, 0]))
            scipy.misc.imsave(outpath.format('tgt', batch_idx, i), map_segmentation_to_colors(target[i, :, :], dataset))
            scipy.misc.imsave(outpath.format('prd', batch_idx, i), map_segmentation_to_colors(output[i, :, :], dataset))
