import pickle
import lmdb
import os
import glob
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np


def pascal_to_lmdb(root_path, split, lmdb_path):

    NUM_CLASSES = 21

    path_to_split_txt = os.path.join(root_path, 'ImageSets/Segmentation', f'{split}.txt')
    image_paths = []

    with open(path_to_split_txt, "r") as fptr:
        image_paths = [x.strip() for x in fptr.readlines() if x.strip() != '']

    print("Generate LMDB to %s" % lmdb_path)

    pixels = 0

    for image_name in tqdm(image_paths):
        image_size = Image.open(os.path.join(root_path, 'JPEGImages', f'{image_name}.jpg')).size
        pixels += image_size[0] * image_size[1]

    print("Pixels in split: ", pixels)

    map_size = pixels * 4 + 100 * 1024 * 1024

    print("Estimated Size: ", map_size)

    isdir = os.path.isdir(lmdb_path)

    db = lmdb.open(lmdb_path, subdir=isdir, map_size=map_size, readonly=False, meminit=False, map_async=True)

    txn = db.begin(write=True)

    key_list = []

    for idx, path in tqdm(enumerate(image_paths)):
        jpg_path = os.path.join(root_path, 'JPEGImages', f'{path}.jpg')
        png_path = os.path.join(root_path, 'SegmentationClassRaw', f'{path}.png')
        image = np.array(Image.open(jpg_path).convert('RGB'), dtype=np.uint8)
        label = np.array(Image.open(png_path), dtype=np.uint8)
        txn.put(u'{}'.format(path).encode('ascii'), pickle.dumps(np.dstack((image, label)), protocol=3))
        key_list.append(path)

    txn.commit()

    keys = [u'{}'.format(k).encode('ascii') for k in key_list]

    with db.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys, protocol=3))
        txn.put(b'__len__', pickle.dumps(len(keys), protocol=3))

    db.sync()
    db.close()


if __name__ == '__main__':
    import sys
    pascal_to_lmdb(sys.argv[1], sys.argv[2], sys.argv[3])
