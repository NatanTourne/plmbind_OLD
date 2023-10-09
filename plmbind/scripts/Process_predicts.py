# imports
import pickle
import argparse
import pandas as pd
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
import torch


parser = argparse.ArgumentParser(
    prog='Process_predicts',
    description='Calculate AUROC and Average Precision')

parser.add_argument("--out_dir", help="Directory for output files", required=True)
parser.add_argument("--TF_info", help="human TF database csv", default = "/home/natant/Thesis-plmbind/Testing_ground/Thesis/info.csv")
parser.add_argument("--Train_TFs_loc",help="Location of pickled list of train TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/train_TFs")
parser.add_argument("--Val_TFs_loc", help="Location of pickled list of validation TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/val_TFs")
parser.add_argument("--Test_TFs_loc", help="Location of pickled list of test TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/test_TFs")


args = parser.parse_args()

if args.out_dir[-1] != "/":
    args.out_dir = args.out_dir+"/"
    
with open(args.Train_TFs_loc, "rb") as f: 
    train_TFs = pickle.load(f)
with open(args.Val_TFs_loc, "rb") as f:
    val_TFs = pickle.load(f)
    
info = pd.read_csv(args.TF_info)
sorted_info = info[[i in train_TFs + val_TFs for i in info["HGNC symbol"]]].reset_index().set_index(["HGNC symbol"]).loc[train_TFs + val_TFs]
settings = pd.read_csv(args.out_dir + "settings.csv")

train_TF_splits = [train_TFs[0:200], train_TFs[200:400], train_TFs[400:]]
train_TF_loss_model_val_DNA_train_TF_AUROC = []
train_TF_loss_model_val_DNA_train_TF_AvP = []
for i in range(3):
    with open(args.out_dir + "train_TF_loss_model_val_DNA_train_TF_part_"+str(i+1)+".pkl", 'rb') as f:
        preds = pickle.load(f)
    AUROC = MultilabelAUROC(num_labels=len(train_TF_splits[i]), average='none')
    AverageP = MultilabelAveragePrecision(num_labels=len(train_TF_splits[i]), average='none')
    train_TF_loss_model_val_DNA_train_TF_AUROC.append(AUROC(torch.cat(list(preds[2])), torch.cat(list(preds[0]))))
    train_TF_loss_model_val_DNA_train_TF_AvP.append(AverageP(torch.cat(list(preds[2])), torch.cat(list(preds[0]))))


with open(args.out_dir + "train_TF_loss_model_val_DNA_val_TF.pkl", 'rb') as f:
    preds = pickle.load(f)
AUROC = MultilabelAUROC(num_labels=len(val_TFs), average='none')
AverageP = MultilabelAveragePrecision(num_labels=len(val_TFs), average='none')
train_TF_loss_model_val_DNA_val_TF_AUROC = AUROC(torch.cat(list(preds[2])), torch.cat(list(preds[0])))
train_TF_loss_model_val_DNA_val_TF_AvP = AverageP(torch.cat(list(preds[2])), torch.cat(list(preds[0])))

train_TF_loss_results = pd.DataFrame({
    "HGNC symbol": train_TFs + val_TFs, 
    "AUROC": list(torch.cat(train_TF_loss_model_val_DNA_train_TF_AUROC).numpy()) + list(train_TF_loss_model_val_DNA_val_TF_AUROC.numpy()),
    "AvP": list(torch.cat(train_TF_loss_model_val_DNA_train_TF_AvP).numpy()) + list(train_TF_loss_model_val_DNA_val_TF_AvP.numpy()),
    "loss_model": ["train_TF_loss"]*len(train_TFs+val_TFs),
    "TF_split": ["train"]*len(train_TFs) + ["val"]*len(val_TFs)})
train_TF_loss_results = pd.concat([settings.iloc[[0]*len(train_TFs+val_TFs)].reset_index(), train_TF_loss_results], axis=1).set_index(["HGNC symbol"])
train_TF_loss_results = pd.concat([train_TF_loss_results, sorted_info], axis = 1)

val_TF_loss_model_val_DNA_train_TF_AUROC = []
val_TF_loss_model_val_DNA_train_TF_AvP = []
for i in range(3):
    with open(args.out_dir + "val_TF_loss_model_val_DNA_train_TF_part_"+str(i+1)+".pkl", 'rb') as f:
        preds = pickle.load(f)
    AUROC = MultilabelAUROC(num_labels=len(train_TF_splits[i]), average='none')
    AverageP = MultilabelAveragePrecision(num_labels=len(train_TF_splits[i]), average='none')
    val_TF_loss_model_val_DNA_train_TF_AUROC.append(AUROC(torch.cat(list(preds[2])), torch.cat(list(preds[0]))))
    val_TF_loss_model_val_DNA_train_TF_AvP.append(AverageP(torch.cat(list(preds[2])), torch.cat(list(preds[0]))))

with open(args.out_dir + "val_TF_loss_model_val_DNA_val_TF.pkl", 'rb') as f:
    preds = pickle.load(f)
AUROC = MultilabelAUROC(num_labels=len(val_TFs), average='none')
AverageP = MultilabelAveragePrecision(num_labels=len(val_TFs), average='none')
val_TF_loss_model_val_DNA_val_TF_AUROC = AUROC(torch.cat(list(preds[2])), torch.cat(list(preds[0])))
val_TF_loss_model_val_DNA_val_TF_AvP = AverageP(torch.cat(list(preds[2])), torch.cat(list(preds[0])))

val_TF_loss_results = pd.DataFrame({
    "HGNC symbol": train_TFs + val_TFs, 
    "AUROC": list(torch.cat(val_TF_loss_model_val_DNA_train_TF_AUROC).numpy()) + list(val_TF_loss_model_val_DNA_val_TF_AUROC.numpy()),
    "AvP": list(torch.cat(val_TF_loss_model_val_DNA_train_TF_AvP).numpy()) + list(val_TF_loss_model_val_DNA_val_TF_AvP.numpy()),
    "loss_model": ["val_TF_loss"]*len(train_TFs+val_TFs),
    "TF_split": ["train"]*len(train_TFs) + ["val"]*len(val_TFs)})
val_TF_loss_results = pd.concat([settings.iloc[[0]*len(train_TFs+val_TFs)].reset_index(), val_TF_loss_results], axis=1).set_index(["HGNC symbol"])
val_TF_loss_results = pd.concat([val_TF_loss_results, sorted_info], axis = 1)

output_df = pd.concat([train_TF_loss_results, val_TF_loss_results])
output_df.to_csv(args.out_dir+"Results.csv")