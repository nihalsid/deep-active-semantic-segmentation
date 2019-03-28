import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import map_segmentations_to_colors, map_segmentation_to_colors
import scipy.misc
import constants
import numpy as np


class TensorboardSummary:

	def __init__(self, directory):
		self.directory = directory


	def create_summary(self):
		writer = SummaryWriter(log_dir=self.directory)
		return writer


	def visualize_image(self, writer, dataset, image, target, output, global_step):
		grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
		writer.add_image('Image', grid_image, global_step)
		grid_image = make_grid(map_segmentations_to_colors(torch.max(output[:3], 1)[1].detach().cpu().numpy(), dataset=dataset), 3, normalize=False, range=(0, 255))
		writer.add_image('Predicted label', grid_image, global_step)
		grid_image = make_grid(map_segmentations_to_colors(torch.squeeze(target[:3], 1).detach().cpu().numpy(), dataset=dataset), 3, normalize=False, range=(0, 255))
		writer.add_image('Groundtruth label', grid_image, global_step)


	@staticmethod
	def visualize_images_to_folder(output_folder, batch_idx, image, target, output, dataset):
		
		for i in range(image.shape[0]):
			outpath = os.path.join(output_folder, '{0}_{1:04d}_{2:02d}.png')
			scipy.misc.imsave(outpath.format('img', batch_idx, i), np.transpose(image[i, :, :, :], axes=[1, 2, 0]))
			scipy.misc.imsave(outpath.format('tgt', batch_idx, i), map_segmentation_to_colors(target[i, :, :], dataset))
			scipy.misc.imsave(outpath.format('prd', batch_idx, i), map_segmentation_to_colors(output[i, :, :], dataset))
