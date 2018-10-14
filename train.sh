# vgg unet
# python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
   # --batch-size 1 --save-every 0 --lambd 10. --train --model vggunet \
#    --sample 3 --no-mse

# vgg unet
# python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
   # --batch-size 1 --save-every 10 --lambd 100 --train --model resunet \
#    --sample 3 --no-mse

# res unet
python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
      --batch-size 1 --save-every 11 --lambd 10 --train --model resunet \
      --sample 3 --no-mse --resblock

# resgen
# python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
    # --batch-size 1 --save-every 11 --lambd 10 --train --model resgen \
#     --sample 3 --no-mse --resblock
