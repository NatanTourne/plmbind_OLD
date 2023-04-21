#!/bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")

current_dir="/home/data/shared/natant/Results/emb480_LR_test_WZ"
mkdir "${current_dir}"
pipeline_dir="/home/natant/Thesis-plmbind/Thesis/plmbind/scripts/Pipeline"


name="LR_0_00001"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_model2_0.py" --out_dir "${current_dir}/${name}" --emb t12_480_pad_trun --emb_dim 480 --num_DNA_filters 60 --learning_rate 0.00001
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb t12_480_pad_trun --emb_dim 480 --num_DNA_filters 60
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"

rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_*"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_*"