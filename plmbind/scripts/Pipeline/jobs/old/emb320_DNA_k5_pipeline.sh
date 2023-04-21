#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")
current_dir="/home/data/shared/natant/Results/FullModel-esm320_DNA_K5"
pipeline_dir="/home/natant/Thesis-plmbind/Thesis/plmbind/scripts/Pipeline"
mkdir "${current_dir}"

name="DNA_branch_small"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --DNA_kernel_size 5 --num_DNA_filters 42
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --DNA_kernel_size 5 --num_DNA_filters 42
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"

name="DNA_branch_medium"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --DNA_kernel_size 5 --num_DNA_filters 84
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --DNA_kernel_size 5 --num_DNA_filters 84
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"

name="DNA_branch_large"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --DNA_kernel_size 5 --num_DNA_filters 124
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_DNA --DNA_kernel_size 5 --num_DNA_filters 124
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"
