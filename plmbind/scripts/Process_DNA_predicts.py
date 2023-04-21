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
parser.add_argument("--TF_info", help="human TF database csv", default = "/home/natant/Thesis-plmbind/Testing_ground/info.csv")
parser.add_argument("--Train_TFs_loc",help="Location of pickled list of train TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/train_TFs")

args = parser.parse_args()

if args.out_dir[-1] != "/":
    args.out_dir = args.out_dir+"/"
    
with open(args.Train_TFs_loc, "rb") as f: 
    train_TFs = pickle.load(f)
    
info = pd.read_csv(args.TF_info)
sorted_info = info[[i in train_TFs for i in info["HGNC symbol"]]].reset_index().set_index(["HGNC symbol"]).loc[train_TFs]
settings = pd.read_csv(args.out_dir + "settings.csv")


with open(args.out_dir + "DNA_model_val_DNA_train_TF.pkl", 'rb') as f:
    preds = pickle.load(f)
AUROC = MultilabelAUROC(num_labels=len(train_TFs), average='none')
AverageP = MultilabelAveragePrecision(num_labels=len(train_TFs), average='none')
DNA_model_val_DNA_train_TF_AUROC = AUROC(torch.cat(list(preds[2])), torch.cat(list(preds[0])))
DNA_model_val_DNA_train_TF_AvP = AverageP(torch.cat(list(preds[2])), torch.cat(list(preds[0])))


DNA_model_results = pd.DataFrame({
    "HGNC symbol": train_TFs, 
    "AUROC": list(DNA_model_val_DNA_train_TF_AUROC.numpy()),
    "AvP": list(DNA_model_val_DNA_train_TF_AvP.numpy()),
    "loss_model": ["DNA_model"]*len(train_TFs),
    "TF_split": ["train"]*len(train_TFs)})
DNA_model_results = pd.concat([settings.iloc[[0]*len(train_TFs)].reset_index(), DNA_model_results], axis=1).set_index(["HGNC symbol"])
DNA_model_results = pd.concat([DNA_model_results, sorted_info], axis = 1)

DNA_model_results.to_csv(args.out_dir+"Results.csv")