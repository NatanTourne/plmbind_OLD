#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")

current_dir="/home/data/shared/natant/Results/DNA_model"
pipeline_dir="/home/natant/Thesis-plmbind/Thesis/plmbind/scripts/Pipeline"

name="small"
mkdir "${current_dir}/${name}"
python "${pipeline_dir}/Train_DNA_model2_0.py" --out_dir  "${current_dir}/${name}" --num_DNA_filters 30 --learning_rate 0.0001

name="medium"
mkdir "${current_dir}/${name}"
python "${pipeline_dir}/Train_DNA_model2_0.py" --out_dir  "${current_dir}/${name}" --num_DNA_filters 60 --learning_rate 0.0001

name="large"
mkdir "${current_dir}/${name}"
python "${pipeline_dir}/Train_DNA_model2_0.py" --out_dir  "${current_dir}/${name}" --num_DNA_filters 90 --learning_rate 0.0001


