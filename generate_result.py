import torch

import os
import random

from PIL import Image
from torchvision import transforms
from preprocess import re_scale, save_image

from models import DeepUNetPaintGenerator
from utils import load_checkpoints

from preprocess import PairedDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_root = './data/test_result'
if not os.path.exists(out_root):
    os.mkdir(out_root)
generator = 'deepunetG_015.pth.tar'

model = DeepUNetPaintGenerator()
for param in model.parameters():
    param.requires_grad = False
model = model.to(device)
load_checkpoints(generator, model)

val_data = PairedDataset(
    transform=transforms.ToTensor(),
    mode='val',
    color_histogram=True,
)
length = len(val_data)
idxs = random.sample(range(0, length - 1), 3400)

targets = idxs[0:1700]
styles = idxs[1700:3400]

to_pil = transforms.ToPILImage()

for i, (target, style) in enumerate(zip(targets, styles)):
    print("Processing %d" % i)
    imageA, imageB, _ = val_data[target]
    styleA, styleB, colors = val_data[style]

    imageA, imageB = imageB, imageA
    styleA, styleB = styleB, styleA

    imageA = imageA.unsqueeze(0).to(device)
    imageB = imageB.unsqueeze(0).to(device)
    styleB = styleB.unsqueeze(0).to(device)
    colors = colors.unsqueeze(0).to(device)

    with torch.no_grad():
        fakeB, _ = model(
            imageA,
            colors,
        )

    result = Image.new('RGB', (4 * 512, 512))
    styleB = styleB.squeeze()
    fakeB = fakeB.squeeze()
    imageA = imageA.squeeze()
    imageB = imageB.squeeze()

    imageA = to_pil(re_scale(imageA).detach().cpu())
    imageB = to_pil(re_scale(imageB).detach().cpu())
    styleB = to_pil(re_scale(styleB).detach().cpu())
    fakeB = to_pil(re_scale(fakeB).detach().cpu())

    result.paste(imageA, (0, 0))
    result.paste(styleB, (512, 0))
    result.paste(fakeB, (512 * 2, 0))
    result.paste(imageB, (512 * 3, 0))

    save_image(result, 'result_%03d' % i, out_root)
