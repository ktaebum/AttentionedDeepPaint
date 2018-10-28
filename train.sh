# vgg unet
# python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
   # --batch-size 1 --save-every 50 --lambd 10. --train --model vggunet \
#    --sample 3 --no-mse --norm instance --num-epochs 300

# res unet
# python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
      # --batch-size 1 --save-every 11 --lambd 10 --train --model resunet \
#       --sample 3 --no-mse --resblock

# resgen
# python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
    # --batch-size 1 --save-every 11 --lambd 10 --train --model resgen \
#     --sample 3 --no-mse --resblock

# python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
      # --batch-size 4 --save-every 10 --lambd 100 --train --model style2paint \
      # --sample 3 --no-mse \
      # --pretrainedG style2paintG_190.pth.tar \
#       --pretrainedD style2paintD_190.pth.tar --last-epoch 190

python train.py --learning-rate 0.00025 --beta1 0.5 --verbose \
       --batch-size 4 --save-every 10 --lambd 50 --train --model style2paint \
       --sample 3 --no-mse --norm instance \

# residual training
# python train.py --learning-rate 0.0002 --beta1 0.5 --verbose --alpha 0.3 \
      # --batch-size 1 --save-every 10 --lambd 10 --train --model residual \
#       --sample 3 --no-mse\
