import pickle
import pyarrow as pa
import lmdb
import os
import glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np


def cityscapes_to_lmdb(root_path, split, lmdb_path):
	
	NUM_CLASSES = 19
	void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
	valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
	# class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
	ignore_index = 255
	assert(len(valid_classes) == NUM_CLASSES)
	class_map = dict(zip(valid_classes + void_classes, list(range(NUM_CLASSES)) + [ignore_index] * len(void_classes)))
	images_base = os.path.join(root_path, 'leftImg8bit', split)
	labels_base = os.path.join(root_path, 'gtFine_trainvaltest', 'gtFine', split)
	image_paths = glob.glob(os.path.join(images_base, '**', '*.png'), recursive=True)
	label_paths = []
	
	for img_path in image_paths:
		label_paths.append(os.path.join(labels_base, Path(img_path).parts[-2], f'{os.path.basename(img_path)[:-15]}gtFine_labelIds.png'))

	print("Generate LMDB to %s" % lmdb_path)
	image_size = Image.open(image_paths[0]).size
	map_size = (len(image_paths) + 10) * image_size[0] * image_size[1] * 4
	print("Estimated Size: ", map_size)
	isdir = os.path.isdir(lmdb_path)
	db = lmdb.open(lmdb_path, subdir=isdir, map_size=map_size, readonly=False, meminit=False, map_async=True)
	txn = db.begin(write=True)
	
	key_list = []

	for idx, datasample in tqdm(enumerate(zip(image_paths, label_paths))):
		image = np.array(Image.open(datasample[0]).convert('RGB'), dtype=np.uint8)
		label = np.array(Image.open(datasample[1]), dtype=np.uint8)
		mapper = np.vectorize(lambda l: class_map[l])
		label[:, :] = mapper(label)
		path_to_image = "/".join(datasample[0].replace(root_path,'').split(os.path.sep))
		txn.put(u'{}'.format(path_to_image).encode('ascii'), pickle.dumps(np.dstack((image, label)), protocol=3))
		key_list.append(path_to_image)

	txn.commit()
	
	keys = [u'{}'.format(k).encode('ascii') for k in key_list]

	with db.begin(write=True) as txn:
		txn.put(b'__keys__', pickle.dumps(keys, protocol=3))
		txn.put(b'__len__', pickle.dumps(len(keys), protocol=3))

	db.sync()
	db.close()


if __name__ == '__main__':
	cityscapes_to_lmdb(r'D:\nihalsid\DeeplabV3+\datasets\cityscapes\\', 'test', r'D:\nihalsid\DeeplabV3+\datasets\cityscapes\test.db')