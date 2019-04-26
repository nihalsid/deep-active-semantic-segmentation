from torchvision import transforms
from dataloaders import custom_transforms as tr
from torch.utils import data
import pickle
from PIL import Image

class PathsDataset(data.Dataset):

		def __init__(self, env, paths, crop_size):

			self.env = env
			self.paths = paths
			self.crop_size = crop_size


		def __len__(self):
			return len(self.paths)


		def __getitem__(self, index):
			
			img_path = self.paths[index]
			loaded_npy = None
			with self.env.begin(write=False) as txn:
				loaded_npy = pickle.loads(txn.get(img_path))
			
			image = loaded_npy[:, :, 0:3]
			
			composed_tr = transforms.Compose([
				tr.FixScaleCropImageOnly(crop_size=self.crop_size),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			])

			return composed_tr(Image.fromarray(image))

