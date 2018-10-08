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

Sketch - Output - Ground Truth
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
    `beta1`  | `0.5`
    `batch_size`  | `4`
    `lambda`  | `10`
    `epochs`  | `200`
    `learning_rate schedule`  | `None`

### Problem
  - Cannot use arbitrary hint

### Examples

Sketch - Output - Ground Truth
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
    `beta1`  | `0.5`
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

