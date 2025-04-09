GPU=$1
DATA=$2

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python pretrain.py \
--dataset ${DATA} \
--epoch_train 500 \
--lr_net 0.01 \
--model ConvNet3D \
--eval_mode SS \
--num_workers 4 \
--preload
