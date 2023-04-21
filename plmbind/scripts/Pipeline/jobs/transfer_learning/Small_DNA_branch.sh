#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")

current_dir="/home/data/shared/natant/Results/DNA_model_transfer"
mkdir "${current_dir}"
pipeline_dir="/home/natant/Thesis-plmbind/Thesis/plmbind/scripts/Pipeline"

name="DNA_small"
mkdir "${current_dir}/${name}"
python "${pipeline_dir}/Train_DNA_model2_0.py" --out_dir  "${current_dir}/${name}" --num_DNA_filters 30 --learning_rate 0.0001
python "${pipeline_dir}/Process_DNA_model2_0.py" --out_dir  "${current_dir}/${name}" --num_DNA_filters 30
python "${pipeline_dir}/Process_DNA_predicts.py" --out_dir "${current_dir}/${name}"
rm "${current_dir}/${name}/DNA_model_val_DNA_"*

name="320_small"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --num_DNA_filters 30 --pre_trained_DNA_branch "${current_dir}/DNA_small/DNA-model-"*"-epoch"*.ckpt
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --num_DNA_filters 30
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"

rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_"*
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_"*