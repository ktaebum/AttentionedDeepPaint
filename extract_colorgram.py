"""
For all datasets, extract top-4 color histogram and save it into json files
"""

import json
import os
import glob

from PIL import Image, ImageOps
from colorgram import colorgram as cgm

out_path = './data/pair_niko/colorgram'
resize_path = './data/pair_niko/resize'
if not os.path.exists(out_path):
    os.mkdir(out_path)
if not os.path.exists(resize_path):
    os.mkdir(resize_path)

data_path = './data/pair_niko'

img_files = glob.glob(os.path.join(data_path, 'train/*.png'))
img_files += glob.glob(os.path.join(data_path, 'val/*.png'))

topk = 4


def get_rgb(colorgram_result):
    """
    from colorgram_result, result rgb value as tuple of (r,g,b)
    """
    color = colorgram_result.rgb
    return (color.r, color.g, color.b)


for filename in img_files:
    image = Image.open(filename)
    width, height = image.size
    image = image.crop((0, 0, width // 2, height))

    image_id = filename.split('/')[-1][:-4]

    # resize
    out_file = os.path.join(resize_path, '%s.png' % image_id)
    image = image.resize((224, 224))
    image.save(out_file)

    # get json
    out_file = os.path.join(out_path, '%s.json' % image_id)
    if os.path.exists(out_file):
        # for continuation
        print('Already processed %s' % image_id)
        continue
    print('processing %s...' % image_id)

    try:
        colors = cgm.extract(image, topk + 1)

        color_dict = {'%d' % i: get_rgb(colors[i]) for i in range(1, topk + 1)}
        with open(out_file, 'w') as json_file:
            json_file.write(json.dumps(color_dict))
    except IndexError:
        print('Remove %s' % filename)
        os.remove(filename)
