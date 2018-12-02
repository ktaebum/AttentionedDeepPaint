python train.py --learning-rate 0.0002 --beta1 0.5 --verbose \
       --batch-size 4 --save-every 5 --lambd 100 --model deepunet \
       --sample 4 --no-mse --norm batch --num-epochs 15 --print-every 300 \
       --train --pretrainedG deepunetG_015.pth.tar --pretrainedD deepunetD_015.pth.tar \
       --last-epoch 15