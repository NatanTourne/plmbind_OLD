#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")

current_dir="/home/data/shared/natant/Results/DL2_0"


name="emb320"
mkdir "${current_dir}"
mkdir "${current_dir}/${name}"

python Train_model2_0.py --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --num_DNA_filters 60
python Process_model2_0.py --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --num_DNA_filters 60
python Process_predicts.py --out_dir "${current_dir}/${name}"