# vgg unet
python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
  --batch-size 4 --save-every 0 --lambd 100. --train --model vggunet \
  --sample 3
