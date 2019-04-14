import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import constants


class Decoder(nn.Module):

	def __init__(self, num_classes, backbone, batchnorm, mc_dropout):

		super(Decoder, self).__init__()

		if backbone == 'resnet':
			low_level_inplanes = 256
		elif backbone == 'xception':
			low_level_inplanes = 128
		elif backbone == 'mobilenet':
			low_level_inplanes = 24
		else:
			raise NotImplementedError

		self.mc_dropout = mc_dropout
		self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
		self.bn1 = batchnorm(48)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout2d(constants.MC_DROPOUT_RATE)

		# aspp always gives out 256 planes + 48 from conv1
		self.last_conv = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
									   batchnorm(256),
									   nn.ReLU(),
									   nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
									   batchnorm(256),
									   nn.ReLU(),
									   nn.Dropout2d(0.1),
									   nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
		self._init_weight()


	def forward(self, x, low_level_feat):

		low_level_feat = self.conv1(low_level_feat)
		low_level_feat = self.bn1(low_level_feat)
		low_level_feat = self.relu(low_level_feat)
		if self.mc_dropout:
			low_level_feat = self.dropout(low_level_feat)

		x = F.interpolate(x, low_level_feat.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x, low_level_feat), dim=1)
		x = self.last_conv(x)

		return x


	def _init_weight(self):
		
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, SynchronizedBatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

if __name__ == '__main__':
	low_level_feat = torch.rand(1, 24, 128, 128)
	input = torch.rand(1, 256, 32, 32)
	model = Decoder(num_classes=19, backbone='mobilenet', batchnorm=nn.BatchNorm2d, mc_dropout=True)
	output = model(input, low_level_feat)
	print(output.size())