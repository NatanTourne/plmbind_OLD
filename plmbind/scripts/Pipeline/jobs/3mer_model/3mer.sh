#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")

current_dir="/home/data/shared/natant/Results/3mer_model"
mkdir "${current_dir}"
pipeline_dir="/home/natant/Thesis-plmbind/Thesis/plmbind/scripts/Pipeline"


name="medium"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_kmer_model2_0.py" --out_dir "${current_dir}/${name}" --emb 3mer_pad_trun --num_DNA_filters 60
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb 3mer_pad_trun --num_DNA_filters 60
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"

rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_"*
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_"*



name="large"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_kmer_model2_0.py" --out_dir "${current_dir}/${name}" --emb 3mer_pad_trun --num_DNA_filters 90
python "${pipeline_dir}/Process_model2_0.py" --out_dir "${current_dir}/${name}" --emb 3mer_pad_trun --num_DNA_filters 90
python "${pipeline_dir}/Process_predicts.py" --out_dir "${current_dir}/${name}"

rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_"*
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_"*