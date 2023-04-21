#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")
current_dir="/home/natant/Thesis-plmbind/Testing_ground/testing_DL2_0/test_module"
pipeline_dir="/home/natant/Thesis-plmbind/Thesis/plmbind/scripts/Pipeline"
mkdir "${current_dir}"

name="pretrained"
mkdir "${current_dir}/${name}"

python "${pipeline_dir}/Train_model2_0.py" --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --num_DNA_filters 30 --pre_trained_DNA_branch /home/natant/Thesis-plmbind/Testing_ground/testing_DL2_0/test_module/DNA-model-20230413_14:47:06-epoch=00-val_loss=0.69.ckpt