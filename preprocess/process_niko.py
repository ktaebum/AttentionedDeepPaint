"""
Preprocess niko dataset
and generate pair image
"""
import os
import glob

from . import get_sketch, save_image

from PIL import Image

if __name__ == "__main__":
    img_files = glob.glob('./data/niko/**/*.jpg', recursive=True)
    for file in img_files:
        filename = file.split('/')[-1][:-4]
        print('Processing %s' % file)
        original = Image.open(file)
        sketch = get_sketch(file, 'more')

        width, height = sketch.size

        concat = Image.new('RGB', (2 * width, height))

        concat.paste(original, (0, 0))
        concat.paste(sketch, (width, 0))

        save_image(concat, filename, './data/pair_niko')
