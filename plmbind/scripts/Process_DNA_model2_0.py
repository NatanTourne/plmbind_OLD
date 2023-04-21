# imports
import pytorch_lightning as pl
from datetime import datetime
import pickle
import argparse
import os
import h5torch
import torch
import numpy as np

# Own imports
from plmbind.data import ReMapDataModule2_0
from plmbind.models import MultilabelModel


# arg parser
parser = argparse.ArgumentParser(
    prog='Process_model',
    description='Predict')

parser.add_argument("--out_dir", help="Dictory for output files", required=True)
parser.add_argument("--emb", help="Protein embeddings to use", default = "t6_320_pad_trun")
parser.add_argument("--window_size", help="The window size", type=int, default=2**16)
parser.add_argument("--batch_size", help="The batch size", type=int, default=16)
parser.add_argument("--num_DNA_filters", type=int, default=50)
parser.add_argument("--DNA_kernel_size", type=int, default=10)
parser.add_argument("--DNA_dropout", type=float, default=0.25)


parser.add_argument("--Train_TFs_loc",help="Location of pickled list of train TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/train_TFs")
parser.add_argument("--Val_TFs_loc", help="Location of pickled list of validation TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/val_TFs")
parser.add_argument("--Test_TFs_loc", help="Location of pickled list of test TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/test_TFs")
parser.add_argument("--train_loc", help="Location of training data",default ="/home/data/shared/natant/Data/train_no_alts.h5t")
parser.add_argument("--val_loc", help="Location of validation data", default ="/home/data/shared/natant/Data/val_no_alts.h5t")
parser.add_argument("--test_loc", help="Location of test data", default ="/home/data/shared/natant/Data/test_no_alts.h5t")

args = parser.parse_args()

# Create unique date timestamp
date = datetime.now().strftime("%Y%m%d_%H:%M:%S")

if args.out_dir[-1] != "/":
    args.out_dir = args.out_dir+"/"
    
Embeddings = "unstructured/" + args.emb

# Load list of TFs used for training (embeddings will be fetched from dataloader)
with open(args.Train_TFs_loc, "rb") as f: 
    train_TFs = pickle.load(f)
with open(args.Val_TFs_loc, "rb") as f:
    val_TFs = pickle.load(f)

# Create datamodule: 
    # Seperate files for train, val, test
    # Protein embeddings are now specified (multiple sizes are possible)
remap_datamodule = ReMapDataModule2_0(
    train_loc=args.train_loc,
    val_loc=args.val_loc,
    test_loc=args.test_loc,
    TF_list=train_TFs,
    val_list=val_TFs,
    window_size=args.window_size,
    embeddings=Embeddings,
    batch_size=args.batch_size
    ) 

# Create Trainer
trainer = pl.Trainer(
    max_epochs = 10000, 
    accelerator = "gpu", 
    devices = [0]
    )

for file in os.listdir(args.out_dir):
    if "DNA-model" in file:
        DNA_model_loc = args.out_dir+file

##### val_TF_loss_model ####
DNA_model = MultilabelModel.load_from_checkpoint(DNA_model_loc)

# Processing Train TFs
remap_datamodule.predict_setup(train_TFs, "val")

preds = list(zip(*trainer.predict(DNA_model, datamodule=remap_datamodule)))
with open(args.out_dir+'DNA_model_val_DNA_train_TF.pkl', 'wb') as f:
    pickle.dump(preds, f)

