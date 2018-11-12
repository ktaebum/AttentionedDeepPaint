# Experiment Results


## Pix2Pix Application
### Configuration
  - Using original pix2pix architecture. 
    - Simple modification to meet 512x512 resolution. 
  - Train Setting
  
    Hyperparameter   | Value
    --------------   | ---------
    `learning_rate`  | `0.0002`
    `beta1`  | `0.5`
    `batch_size`  | `4`
    `lambda`  | `100`
    `epochs`  | `200`
    `learning_rate schedule`  | `None`

### Problem
  - Do not use style hint
  - Biased pattern appeared

### Examples

Sketch - Ground Truth - Output
![pix2pix1](https://i.imgur.com/swgvRAl.png)
## VGGUNet-concat Application
### Configuration
  - Using VGGNet19_bn as feature extractor as style2paint paper
  - Add 2 guide decoder
  - Use ground truth as style hint
  - concat extracted feature (not add)
  - Train Setting
  
    Hyperparameter   | Value
    --------------   | ---------
    `learning_rate`  | `0.0002`
    `beta1 (optimizer)`  | `0.5`
    `alpha (loss)`  | `0.3`
    `beta (loss)`  | `0.9`
    `batch_size`  | `4`
    `lambda`  | `10`
    `epochs`  | `200`
    `learning_rate schedule`  | `None`

### Problem
  - Cannot use arbitrary hint

### Examples

Sketch - Ground Truth - Output
![vggconcat1](https://i.imgur.com/OXGhdqO.png)
![vggconcat2](https://i.imgur.com/EPvoVsY.png)
![vggconcat3](https://i.imgur.com/fnZKQfE.png)


## VGGUNet-add Application
### Configuration
  - Using VGGNet19_bn as feature extractor as style2paint paper
  - Add 2 guide decoder
  - Use ground truth as style hint in training and use arbitrary style in inference
  - add extracted feature (not concat)
  - Train Setting
  
    Hyperparameter   | Value
    --------------   | ---------
    `learning_rate`  | `0.0002`
    `beta1 (optimizer)`  | `0.5`
    `alpha (loss)`  | `0.3`
    `beta (loss)`  | `0.9`
    `batch_size`  | `1`
    `lambda`  | `10`
    `epochs`  | `200`
    `learning_rate schedule`  | `None`

### Problem
  - Colorize well, but cannot apply hint well yet...

### Examples
Sketch - Style - Output - Ground Truth 
![vggadd1](https://i.imgur.com/l2reucJ.png)
![vggadd2](https://i.imgur.com/vUMMc2u.png)
![vggadd3](https://i.imgur.com/9t9JFTC.png)


## Style2Paint Application
### Configuration
  - Using Style2Paint Original Paper (But using patch-gan discriminator)
  - Add 2 guide decoder
  - Train Setting

    Hyperparameter   | Value
    --------------   | ---------
    `learning_rate`  | `0.0002`
    `beta1 (optimizer)`  | `0.5`
    `alpha (loss)`  | `0.3`
    `beta (loss)`  | `0.9`
    `batch_size`  | `4`
    `lambda`  | `50`
    `epochs`  | `200`
    `learning_rate schedule`  | `None`

### Problem
  - Still, cannot import style reference well...
  - Noisy background appear
  - How to solve :(

### Examples
Sketch - Style - Guide1 - Guide2 - Output - Ground Truth
![style2paint1](https://i.imgur.com/VbIceI6.png)
![style2paint2](https://i.imgur.com/QXgcFgC.png)
![style2paint3](https://i.imgur.com/waWRxUY.png)
![style2paint4](https://i.imgur.com/RbAafXs.png)

## DeepPaint Application
### Configuration
  - Our idea (Using SegUnet with Colorgram)
  - Use VGGNet19_bn to extract feature
  - Used 2 guide decoder
  - Train Setting

    Hyperparameter   | Value
    --------------   | ---------
    `learning_rate`  | `0.0002`
    `beta1 (optimizer)`  | `0.5`
    `alpha (loss)`  | `0.3`
    `beta (loss)`  | `0.9`
    `batch_size`  | `4`
    `lambda`  | `150`
    `epochs`  | `200`
    `learning_rate schedule`  | `None`

### Problem
  - It converts style much better than previous models
  - But still cannot get style for dark image
  - Cannot colorize well for noisy background image
  - How to notice portion of color?
  - How about using attention mechanism
  - More deep generator and more deep discriminator
  - Using different feature extraction model (for tag prediction model)

### Examples
Sketch - Style - Guide1 - Guide2 - Output - Ground Truth - Colorgram of Style
![DeepPaint1](https://i.imgur.com/VTzu8H2.png)
![DeepPaint2](https://i.imgur.com/wS3W8L0.png)
![DeepPaint3](https://i.imgur.com/xNMnMdl.png)
![DeepPaint4](https://i.imgur.com/Bq0fr1X.png)
![DeepPaint5](https://i.imgur.com/XmvYmLn.png)
![DeepPaint6](https://i.imgur.com/FATdxLE.png)
