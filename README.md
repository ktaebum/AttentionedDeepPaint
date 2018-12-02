# AttentionedDeepPaint

## Automatic Anime Sketch Colorization Using Generative Adversarial Network (GAN)

## Result Examples

![refer](https://i.imgur.com/lDzhfjK.png)
![Result1](https://i.imgur.com/ls1gjNY.png)
![Result2](https://i.imgur.com/C49CdDC.png)
![Result3](https://i.imgur.com/lzHKnm0.png)
![Result4](https://i.imgur.com/kwMmfWp.png)
![Result5](https://i.imgur.com/bUEzmI6.png)
![Result6](https://i.imgur.com/6U1VWtL.png)
![Result7](https://i.imgur.com/q5Tr7y8.png)

To see experiment trials, [click](https://github.com/ktaebum/AttentionedDeepPaint/tree/master/results)

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

Maybe it can be trained more...?
