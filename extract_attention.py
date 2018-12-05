import torch

import os
import random

import skimage.transform

from PIL import Image
from torchvision import transforms
from preprocess import re_scale, save_image

from models import DeepUNetPaintGenerator
from utils import load_checkpoints

from preprocess import PairedDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

out_root = './data/attention_result'
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
        fakeB, attentions = model(
            imageA,
            colors,
        )

    def interpolate(image):
        image = image.squeeze().detach().cpu().numpy()
        if image.shape[-1] != 512:
            image = skimage.transform.pyramid_expand(
                image,
                upscale=512 // (image.shape[-1]),
                multichannel=False,
            )
        return to_pil(torch.Tensor(image).unsqueeze(0))

    attentions = list(map(lambda img: interpolate(img), attentions))

    result = Image.new('RGBA', (4 * 512, 512))
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

    figure = Image.new('RGB', (9 * 512, 512))
    figure.paste(imageA, (0, 0))
    figure.paste(styleB, (512 * 1, 0))
    figure.paste(attentions[0], (512 * 2, 0))
    figure.paste(attentions[1], (512 * 3, 0))
    figure.paste(attentions[2], (512 * 4, 0))
    figure.paste(attentions[3], (512 * 5, 0))
    figure.paste(attentions[4], (512 * 6, 0))
    figure.paste(fakeB, (512 * 7, 0))
    figure.paste(imageB, (512 * 8, 0))
    save_image(figure, 'attention_%03d' % i, out_root)
