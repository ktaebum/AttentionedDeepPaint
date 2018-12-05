"""
For all datasets, extract top-4 color histogram and save it into json files
"""

import json
import os
import glob

from PIL import Image
from colorgram import colorgram as cgm

data_path = './data'
out_path = './data/colorgram'

img_files = glob.glob(os.path.join(data_path, 'train/*.png'))
img_files += glob.glob(os.path.join(data_path, 'val/*.png'))

topk = 4


def get_rgb(colorgram_result):
    """
    from colorgram_result, result rgb value as tuple of (r,g,b)
    """
    color = colorgram_result.rgb
    return (color.r, color.g, color.b)


def crop_region(image):
    """
    from image, crop 4 region and return
    """
    width, height = image.size
    h1 = height // 4
    h2 = h1 + h1
    h3 = h2 + h1
    h4 = h3 + h1
    image1 = image.crop((0, 0, width, h1))
    image2 = image.crop((0, h1, width, h2))
    image3 = image.crop((0, h2, width, h3))
    image4 = image.crop((0, h3, width, h4))

    return (image1, image2, image3, image4)


def get_topk(color_info, k):
    colors = list(color_info.values())
    return list(map(lambda x: x[k], colors))


for filename in img_files:
    image = Image.open(filename)
    width, height = image.size
    image = image.crop((0, 0, width // 2, height))

    image_id = filename.split('/')[-1][:-4]

    # get json
    out_file = os.path.join(out_path, '%s.json' % image_id)
    if os.path.exists(out_file):
        # for continuation
        print('Already processed %s' % image_id)
        continue
    print('processing %s...' % image_id)

    try:
        images = list(crop_region(image))
        result = {}
        for i, img in enumerate(images, 1):
            colors = cgm.extract(img, topk + 1)
            result[str(i)] = {
                '%d' % i: get_rgb(colors[i])
                for i in range(1, topk + 1)
            }
        with open(out_file, 'w') as json_file:
            json_file.write(json.dumps(result))

    except IndexError:
        print('Remove %s' % filename)
        os.remove(filename)
