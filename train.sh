# pix2pix
# python pix2pix.py --learning-rate 0.0002 --beta1 0.5 --mode B2A --verbose \
  # --batch-size 4 --train --save-every 10 --lambd 100

# vgg unet
python vggunet.py --learning-rate 0.0002 --beta1 0.5 --mode B2A --verbose \
  --batch-size 4 --save-every 10 --lambd 100 --train
