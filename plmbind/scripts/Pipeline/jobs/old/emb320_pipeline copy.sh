#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")
current_dir="/home/data/shared/natant/Results/FullModel-esm320"
pipeline_dir="/home/natant/Thesis-plmbind/Thesis/plmbind/scripts/Pipeline"
mkdir "${current_dir}"

name="DNA_branch_small"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --num_DNA_filters 30
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --num_DNA_filters 30
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"

name="DNA_branch_medium"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --num_DNA_filters 60
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --num_DNA_filters 60
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"

name="DNA_branch_large"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --num_DNA_filters 90
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --num_DNA_filters 90
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"