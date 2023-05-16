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
parser.add_argument("--Val_TFs_loc", help="Location of pickled list of validation TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/val_TFs")
parser.add_argument("--Test_TFs_loc", help="Location of pickled list of test TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/test_TFs")


args = parser.parse_args()

if args.out_dir[-1] != "/":
    args.out_dir = args.out_dir+"/"
    

with open(args.Test_TFs_loc, "rb") as f:
    test_TFs = pickle.load(f)
    
info = pd.read_csv(args.TF_info)
sorted_info = info[[i in test_TFs for i in info["HGNC symbol"]]].reset_index().set_index(["HGNC symbol"]).loc[test_TFs]
settings = pd.read_csv(args.out_dir + "settings.csv")



with open(args.out_dir + "train_TF_loss_model_test_DNA_test_TF.pkl", 'rb') as f:
    preds = pickle.load(f)
AUROC = MultilabelAUROC(num_labels=len(test_TFs), average='none')
AverageP = MultilabelAveragePrecision(num_labels=len(test_TFs), average='none')
train_TF_loss_model_test_DNA_test_TF_AUROC = AUROC(torch.cat(list(preds[2])), torch.cat(list(preds[0])))
train_TF_loss_model_test_DNA_test_TF_AvP = AverageP(torch.cat(list(preds[2])), torch.cat(list(preds[0])))

train_TF_loss_results = pd.DataFrame({
    "HGNC symbol": test_TFs, 
    "AUROC": list(train_TF_loss_model_test_DNA_test_TF_AUROC.numpy()),
    "AvP": list(train_TF_loss_model_test_DNA_test_TF_AvP.numpy()),
    "loss_model": ["train_TF_loss"]*len(test_TFs),
    "TF_split": ["test"]*len(test_TFs)})
train_TF_loss_results = pd.concat([settings.iloc[[0]*len(test_TFs)].reset_index(), train_TF_loss_results], axis=1).set_index(["HGNC symbol"])
train_TF_loss_results = pd.concat([train_TF_loss_results, sorted_info], axis = 1)


with open(args.out_dir + "val_TF_loss_model_test_DNA_test_TF.pkl", 'rb') as f:
    preds = pickle.load(f)
AUROC = MultilabelAUROC(num_labels=len(test_TFs), average='none')
AverageP = MultilabelAveragePrecision(num_labels=len(test_TFs), average='none')
val_TF_loss_model_test_DNA_test_TF_AUROC = AUROC(torch.cat(list(preds[2])), torch.cat(list(preds[0])))
val_TF_loss_model_test_DNA_test_TF_AvP = AverageP(torch.cat(list(preds[2])), torch.cat(list(preds[0])))

val_TF_loss_results = pd.DataFrame({
    "HGNC symbol": test_TFs, 
    "AUROC": list(val_TF_loss_model_test_DNA_test_TF_AUROC.numpy()),
    "AvP": list(val_TF_loss_model_test_DNA_test_TF_AvP.numpy()),
    "loss_model": ["val_TF_loss"]*len(test_TFs),
    "TF_split": ["test"]*len(test_TFs)})
val_TF_loss_results = pd.concat([settings.iloc[[0]*len(test_TFs)].reset_index(), val_TF_loss_results], axis=1).set_index(["HGNC symbol"])
val_TF_loss_results = pd.concat([val_TF_loss_results, sorted_info], axis = 1)

output_df = pd.concat([train_TF_loss_results, val_TF_loss_results])
output_df.to_csv(args.out_dir+"Test_TF_Results.csv")