import os
import random
import glob

train_files = glob.glob('./data/pair_niko/*.png')

whole_files = len(train_files)

val = int(whole_files * 0.2)

random.shuffle(train_files)

val_files = train_files[:val]
train_files = train_files[val:]

for file in train_files:
    filename = file.split('/')[-1]
    os.rename(file, './data/pair_niko/train/%s' % filename)

for file in val_files:
    filename = file.split('/')[-1]
    os.rename(file, './data/pair_niko/val/%s' % filename)

print('train_files', len(train_files))
print('val files', len(val_files))
