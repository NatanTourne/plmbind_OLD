#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")

current_dir="/home/natant/Thesis-plmbind/Testing_ground/testing_DL2_0"
pipeline_dir="/home/natant/Thesis-plmbind/Thesis/plmbind/scripts/Pipeline"

name="test_module"
mkdir "${current_dir}/${name}"
python "${pipeline_dir}/Train_DNA_model2_0.py" --out_dir  "${current_dir}/${name}" --num_DNA_filters 30 --learning_rate 0.0001 --max_epochs 1 --limit_train_batches 10 --limit_val_batches 10