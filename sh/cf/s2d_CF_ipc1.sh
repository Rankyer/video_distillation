GPU=$1
DATA=$2
L_D=$3
L_H=$4

cd ../..;

CUDA_VISIBLE_DEVICES=${GPU} python distill_CF_plus.py \
--method DM \
--dataset ${DATA} \
--num_eval 3 \
--vpc 1 \
--spc 2 \
--dpc 2 \
--epoch_eval_train 500 \
--lr_dynamic=${L_D} \
--lr_hal=${L_H} \
--model=ConvNet3D \
--batch_real 64 \
--Iteration  5000 \
--model ConvNet3D \
--eval_mode SS \
--eval_it 100 \
--no_train_static \
--path_static ./static_memory/miniUCF_spc2.pt \
--startIt 100 \
--preload \
--save_path ./result/ \


