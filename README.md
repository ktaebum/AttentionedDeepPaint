# AttentionedDeepPaint

## Automatic Anime Sketch Colorization Using Generative Adversarial Network (GAN)

## Colorize Result Examples

![refer](https://i.imgur.com/lDzhfjK.png)
![Result2](https://i.imgur.com/C49CdDC.png)
![Result3](https://i.imgur.com/lzHKnm0.png)
![Result4](https://i.imgur.com/kwMmfWp.png)
![Result5](https://i.imgur.com/bUEzmI6.png)
![Result6](https://i.imgur.com/6U1VWtL.png)
![Result7](https://i.imgur.com/q5Tr7y8.png)
![Result8](https://i.imgur.com/1g5f0vE.png)
![Result9](https://i.imgur.com/Rfvf9i4.png)
![Result10](https://i.imgur.com/EpcLh22.png)
![Result11](https://i.imgur.com/IzeXRMb.png)

To see experiment trials, [click](https://github.com/ktaebum/AttentionedDeepPaint/tree/master/results)

## Attention Map Result Examples
![attrefer](https://i.imgur.com/Y1SPOFy.png)
![Att1](https://i.imgur.com/AnqFQul.png)
![Att2](https://i.imgur.com/Gs4yQq8.png)
![Att3](https://i.imgur.com/40ZVsk7.png)
![Att4](https://i.imgur.com/r5FhAwQ.png)
![Att5](https://i.imgur.com/UV40RH3.png)

## Train Settings & Log

Hyperparameter   | Value
--------------   | ---------
`learning_rate`  | `0.0002`
`beta1 (optimizer)`  | `0.5`
`batch_size`  | `4`
`lambda`  | `100`
`epochs`  | `15`
`learning_rate schedule`  | `None`
`Discriminator` | `PatchGAN`
`Weight Initialization` | `(0, 0.02) Normal Distribution`

![Log](https://i.imgur.com/6qwIfXj.png)

It seems that it can be trained more.  
However, if train more, the reconstruction loss increase.  
Based on this weird behavior, I recommand 15 epochs to train (max 20).

## Dependency

Refer [requirements](https://github.com/ktaebum/AttentionedDeepPaint/tree/master/requirements.txt)

Install 
1. Pytorch (>= 0.4.1)
2. Torchvision (>= 0.2.1)
based on your python version, os, cuda version etc...

## Usage

### Download dataset

1. Create **data** folder
2. go to [link](https://www.kaggle.com/ktaebum/animesketchcolorpair) and download
3. unzip in **data** folder

### Train

`
$ ./train.sh
`
