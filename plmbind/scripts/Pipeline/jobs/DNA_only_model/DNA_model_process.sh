#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")

current_dir="/home/data/shared/natant/Results/DNA_model"
pipeline_dir="/home/natant/Thesis-plmbind/Thesis/plmbind/scripts/Pipeline"

name="small"
mkdir "${current_dir}/${name}"
python "${pipeline_dir}/Process_DNA_model2_0.py" --out_dir  "${current_dir}/${name}" --num_DNA_filters 30
python "${pipeline_dir}/Process_DNA_predicts.py" --out_dir "${current_dir}/${name}"
rm "${current_dir}/${name}/DNA_model_val_DNA_"*



name="medium"
mkdir "${current_dir}/${name}"
python "${pipeline_dir}/Process_DNA_model2_0.py" --out_dir  "${current_dir}/${name}" --num_DNA_filters 60
python "${pipeline_dir}/Process_DNA_predicts.py" --out_dir "${current_dir}/${name}"
rm "${current_dir}/${name}/DNA_model_val_DNA_"*



name="large"
mkdir "${current_dir}/${name}"
python "${pipeline_dir}/Process_DNA_model2_0.py" --out_dir  "${current_dir}/${name}" --num_DNA_filters 90
python "${pipeline_dir}/Process_DNA_predicts.py" --out_dir "${current_dir}/${name}"
rm "${current_dir}/${name}/DNA_model_val_DNA_"*

