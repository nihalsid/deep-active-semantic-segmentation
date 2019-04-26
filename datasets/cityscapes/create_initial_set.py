import os
import glob
import random
import sys

path = os.path.normpath(os.path.join('.', 'leftImg8bit', 'train'))
image_paths = glob.glob(os.path.join(path, '**', '*.png'), recursive=True)

indices = random.sample(range(len(image_paths)), 50)

with open(f'set_{sys.argv[1]}.txt', 'w') as fptr:
    for i in indices:
        fptr.write('/' + image_paths[i].replace("\\", "/") + "\n")
