import os
import glob
import random
import sys

path = os.path.normpath(os.path.join('.', 'unprocessed/ImageSets/Segmentation/train.txt'))
with open(path, "r") as fptr:
    image_paths = [x.strip() for x in fptr.readlines() if x.strip() != '']

indices = random.sample(range(len(image_paths)), 50)

with open(f'set_{sys.argv[1]}.txt', 'w') as fptr:
    for i in indices:
        fptr.write(image_paths[i] + "\n")
