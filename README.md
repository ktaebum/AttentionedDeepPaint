# AttentionedDeepPaint

## Automatic Anime Sketch Colorization Using Generative Adversarial Network (GAN)

## Dependency

Refer [requirements](https://github.com/ktaebum/AttentionedDeepPaint/tree/master/requirements.txt)

Install 
1. Pytorch (>= 0.4.1) (No problem in torch 1.0.1)
2. Torchvision (>= 0.2.1)
based on your python version, os, cuda version etc...

## Usage

### Download dataset

1. go to [link](https://www.kaggle.com/ktaebum/anime-sketch-colorization-pair) and download
2. unzip 

### Train

`
$ ./train.sh
`

### Reference

1. Sketch Image Generation
  - https://github.com/lllyasviel/sketchKeras
2. Color Histogram Extraction
  - https://github.com/obskyr/colorgram.py
3. Attentioned UNet
  - https://arxiv.org/pdf/1804.03999.pdf
  - https://github.com/ozan-oktay/Attention-Gated-Networks

## Colorize Result Examples

![refer](https://i.imgur.com/lDzhfjK.png)
![Result1](https://i.imgur.com/F0zuDnY.png)
![Result2](https://i.imgur.com/QiX6GGU.png)
![Result3](https://i.imgur.com/Pi6gFGl.png)
![Result4](https://i.imgur.com/Nm0Tumx.png)
![Result5](https://i.imgur.com/wBoutWk.png)
![Result6](https://i.imgur.com/pBnzZ5x.png)
![Result7](https://i.imgur.com/ZFAv9lr.png)
![Result8](https://i.imgur.com/bItmass.png)
![Result9](https://i.imgur.com/kfNPGdl.png)
![Result10](https://i.imgur.com/QXwZruo.png)
![Result11](https://i.imgur.com/wPyMv5M.png)
![Result12](https://i.imgur.com/7MhZkOl.png)
![Result13](https://i.imgur.com/mCYlIPU.png)
![Result14](https://i.imgur.com/cTatYf5.png)
![Result15](https://i.imgur.com/ibSFmpb.png)
![Result16](https://i.imgur.com/Au3VWJU.png)
![Result17](https://i.imgur.com/kiwREx9.png)

To see experiment trials, [click](https://github.com/ktaebum/AttentionedDeepPaint/tree/master/results)

## General Result Example (Same Sketch, Different Styles)

![Gen1](https://i.imgur.com/Ow8iNAp.png)
![Gen2](https://i.imgur.com/6efKure.png)
![Gen3](https://i.imgur.com/7liDJ0V.png)
![Gen4](https://i.imgur.com/TCwEyfy.png)
![Gen5](https://i.imgur.com/Tj4NEed.png)
![Gen6](https://i.imgur.com/gYz5iPS.png)
![Gen7](https://i.imgur.com/eHlqFuu.png)
![Gen8](https://i.imgur.com/yUWwZw8.png)
![Gen9](https://i.imgur.com/RkRRqDn.png)
![Gen10](https://i.imgur.com/CnLnCfK.png)
![Gen11](https://i.imgur.com/Nq8Yo4i.png)
![Gen12](https://i.imgur.com/4Pjv9zl.png)

## Attention Map Result Examples
![attrefer](https://i.imgur.com/Y1SPOFy.png)
![Att1](https://i.imgur.com/Unu0BBm.png)
![Att2](https://i.imgur.com/D339Ren.png)
![Att3](https://i.imgur.com/n2ZX839.png)
![Att4](https://i.imgur.com/nC9qA2m.png)
![Att5](https://i.imgur.com/MCTYXyf.png)

## Train Settings & Log

Hyperparameter   | Value
--------------   | ---------
`learning_rate`  | `0.0002`
`beta1 (optimizer)`  | `0.5`
`batch_size`  | `4`
`lambda`  | `100`
`epochs`  | `30`
`learning_rate schedule`  | `None`
`Discriminator` | `PatchGAN`
`Weight Initialization` | `(0, 0.02) Normal Distribution`

![Log](https://i.imgur.com/nM60lAM.png)
