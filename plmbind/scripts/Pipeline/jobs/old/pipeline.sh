#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")

current_dir="/home/natant/Thesis-plmbind/Results/20230330_testing_model_sizes"

mkdir "${current_dir}/emb320"
python Train_model.py --out_dir "${current_dir}/emb320" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_TF --num_DNA_filters 60

mkdir "${current_dir}/emb480"
python Train_model.py --out_dir "${current_dir}/emb480" --emb t12_480_pad_trun --emb_dim 480 --early_stop val_loss_TF --num_DNA_filters 60

mkdir "${current_dir}/emb640"
python Train_model.py --out_dir  "${current_dir}/emb640" --emb t30_640_pad_trun --emb_dim 640 --early_stop val_loss_TF --num_DNA_filters 60

