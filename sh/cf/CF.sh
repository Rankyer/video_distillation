GPU=$1
DATA=$2
LR=$3
IPC=$4

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python distill_CF.py \
--method CF \
--dataset ${DATA} \
--ipc ${IPC} \
--num_eval 1 \
--epoch_eval_train 500 \
--init real \
--lr_img ${LR} \
--lr_net 0.01 \
--Iteration 1500 \
--model ConvNet3D \
--eval_mode SS \
--eval_it 50 \
--batch_real 64 \
--num_workers 4 \
--preload
