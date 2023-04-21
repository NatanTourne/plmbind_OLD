#! /bin/bash

now=$(date +"%Y_%m_%d__%H:%M:%S")

current_dir="/home/data/shared/natant/Results/embs"

name="emb320"
mkdir "${current_dir}/${name}"
python Process_model.py --out_dir "${current_dir}/${name}" --emb t6_320_pad_trun --emb_dim 320 --early_stop val_loss_TF --num_DNA_filters 60
python Process_predicts.py --out_dir "${current_dir}/${name}"

rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_train_TF_part_1.pkl"
rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_train_TF_part_2.pkl"
rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_train_TF_part_3.pkl"
rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_val_TF.pkl"

rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_train_TF_part_1.pkl"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_train_TF_part_2.pkl"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_train_TF_part_3.pkl"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_val_TF.pkl"


name="emb480"
mkdir "${current_dir}/${name}"
python Process_model.py --out_dir "${current_dir}/${name}" --emb t12_480_pad_trun --emb_dim 480 --early_stop val_loss_TF --num_DNA_filters 60
python Process_predicts.py --out_dir "${current_dir}/${name}"

rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_train_TF_part_1.pkl"
rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_train_TF_part_2.pkl"
rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_train_TF_part_3.pkl"
rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_val_TF.pkl"

rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_train_TF_part_1.pkl"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_train_TF_part_2.pkl"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_train_TF_part_3.pkl"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_val_TF.pkl"


name="emb640"
mkdir "${current_dir}/${name}"
python Train_model.py --out_dir  "${current_dir}/${name}" --emb t30_640_pad_trun --emb_dim 640 --early_stop val_loss_TF --num_DNA_filters 60
python Process_model.py --out_dir "${current_dir}/${name}" --emb t30_640_pad_trun --emb_dim 640 --early_stop val_loss_TF --num_DNA_filters 60
python Process_predicts.py --out_dir "${current_dir}/${name}"

rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_train_TF_part_1.pkl"
rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_train_TF_part_2.pkl"
rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_train_TF_part_3.pkl"
rm "${current_dir}/${name}/val_TF_loss_model_val_DNA_val_TF.pkl"

rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_train_TF_part_1.pkl"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_train_TF_part_2.pkl"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_train_TF_part_3.pkl"
rm "${current_dir}/${name}/train_TF_loss_model_val_DNA_val_TF.pkl"