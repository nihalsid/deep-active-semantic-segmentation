from torch.utils import data
import glob 
from pathlib import Path


class Cityscapes(data.Dataset):
    
    NUM_CLASSES = 19
    
    def __init__(self, path, split):
    	
    	self.path = path
    	self.split = split

    	self.images_base = os.path.join(self.path, 'leftImg8bit', self.split)
    	self.labels_base = os.path.join(self.path, 'gtFine_trainvaltest', 'gtFine', self.split)

    	self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
    	self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    	self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
    	self.ignore_index = 255
    	self.class_map = dict(zip(self.valid_classes + self.void_classes, range(self.NUM_CLASSES) + [self.ignore_index] * len(self.void_classes)))
		
		self.image_paths = glob.glob(os.path.join(self.images_base, '**', '*.png'), recursive=True)


	def __len__(self):
		return len(self.image_paths)


	def __getitem__(self, index):

		img_path = self.image_paths[index]
		lbl_path = os.path.join(self.labels_base, Path(img_path).parts[-2], f'{os.path.basename(img_path)[:-15]}gtFine_labelIds.png')

		image = Image.open(img_path).convert('RGB')
		target = np.array(Image.open(lbl_path), dtype=np.uint8)

