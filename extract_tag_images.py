import json
import os

from shutil import copyfile

tag_files = ['./tag/white_1girl.json', './tag/white_1boy.json']

images_root = './data/danbour/'
out_root = './data/filtered'


def main():
    for tag_file in tag_files:
        with open(tag_file, 'r') as json_file:
            for line in json_file:
                tags = json.loads(line)
                tag_id = tags['id']
                img_id = tag_id[-3:]
                if int(img_id) > 150:
                    # not in our dataset
                    continue
                else:
                    img_id = '0' + img_id
                    root = os.path.join(images_root, img_id)
                    filename = tag_id + '.jpg'
                    print('processing %s...' % filename)
                    try:
                        copyfile(
                            os.path.join(root, filename),
                            os.path.join(out_root, filename))
                    except FileNotFoundError:
                        continue


if __name__ == "__main__":
    main()
