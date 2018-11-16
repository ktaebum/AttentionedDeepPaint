from PIL import Image
from preprocess import save_image
from preprocess import get_sketch, black2white
import glob

# set image path
img_files = glob.glob('./data/filtered/*.jpg')

for file in img_files:
    filename = file.split('/')[-1]
    print('processing %s...' % filename)
    image = Image.open(file)

    # remove black background
    image = black2white(image, 80)

    width, height = image.size
    colored = image

    sketch = get_sketch(colored, 'more', 1)

    width, height = image.size
    image = Image.new('RGB', (width * 2, height))
    image.paste(colored, (0, 0))
    image.paste(sketch, (width, 0))

    save_image(image, filename[:-4], './data/pair_niko')
