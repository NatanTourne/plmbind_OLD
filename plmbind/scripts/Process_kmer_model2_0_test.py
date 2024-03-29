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
from plmbind.models import PlmbindFullModel_kmer


# arg parser
parser = argparse.ArgumentParser(
    prog='Process_model',
    description='Predict')

parser.add_argument("--out_dir", help="Dictory for output files", required=True)
parser.add_argument("--prot_branch", help="small or big", default="small")
parser.add_argument("--early_stop", help="Early stopping parameter. Either val_loss_DNA or val_loss_TF", default="val_loss_DNA")
parser.add_argument("--emb", help="kmers to use", default = "3mer_pad_trun")
parser.add_argument("--kmer_embedding_size", help="kmers embedding size to use", default=32)
parser.add_argument("--num_kmers", help="The size of the embeddings", type=int, default=8000)
parser.add_argument("--window_size", help="The window size", type=int, default=2**16)
parser.add_argument("--batch_size", help="The batch size", type=int, default=16)
parser.add_argument("--TF_batch_size",  help="The number of TFs to subsample", type=int, default=0)

parser.add_argument("--num_DNA_filters", type=int, default=50)
parser.add_argument("--num_prot_filters", type=int, default=50)
parser.add_argument("--DNA_kernel_size", type=int, default=10)
parser.add_argument("--prot_kernel_size", type=int, default=10)
parser.add_argument("--prot_dropout", type=float, default=0.25)
parser.add_argument("--DNA_dropout", type=float, default=0.25)
parser.add_argument("--latent_vector_size", type=int, default=128)
parser.add_argument("--calculate_val_TF_loss", type=bool, default=True)


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
    TF_batch_size=args.TF_batch_size, # PUT 0 WHEN YOU WANT TO USE ALL TFs
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
    if "Full-kmer-model-train_TF_loss-" in file:
        train_TF_loss_model_loc = args.out_dir+file
    if "Full-kmer-model-val_TF_loss-" in file:
        val_TF_loss_model_loc = args.out_dir+file

##### val_TF_loss_model ####
Full_model = PlmbindFullModel_kmer.load_from_checkpoint(val_TF_loss_model_loc)

    
# Processing Validation TFs
remap_datamodule.predict_setup(val_TFs, "test")
preds = list(zip(*trainer.predict(Full_model, datamodule=remap_datamodule)))
with open(args.out_dir+'val_TF_loss_model_test_DNA_val_TF.pkl', 'wb') as f:
    pickle.dump(preds, f)

# Processing Train TFs
preds = []
train_TF_splits = [train_TFs[0:200], train_TFs[200:400], train_TFs[400:]]
for count, subset in enumerate(train_TF_splits):
    remap_datamodule.predict_setup(subset, "test")
    preds = list(zip(*trainer.predict(Full_model, datamodule=remap_datamodule)))
    with open(args.out_dir+'val_TF_loss_model_test_DNA_train_TF_part_' + str(count+1) + '.pkl', 'wb') as f:
        pickle.dump(preds, f)
    preds = []

remap_datamodule.predict_setup(train_TFs, "test")



Full_model = PlmbindFullModel_kmer.load_from_checkpoint(train_TF_loss_model_loc)

    
# Processing Validation TFs
remap_datamodule.predict_setup(val_TFs, "test")
preds = list(zip(*trainer.predict(Full_model, datamodule=remap_datamodule)))
with open(args.out_dir+'train_TF_loss_model_test_DNA_val_TF.pkl', 'wb') as f:
    pickle.dump(preds, f)

# Processing Train TFs
preds = []
train_TF_splits = [train_TFs[0:200], train_TFs[200:400], train_TFs[400:]]
for count, subset in enumerate(train_TF_splits):
    remap_datamodule.predict_setup(subset, "test")
    preds = list(zip(*trainer.predict(Full_model, datamodule=remap_datamodule)))
    with open(args.out_dir+'train_TF_loss_model_test_DNA_train_TF_part_'+str(count+1)+'.pkl', 'wb') as f:
        pickle.dump(preds, f)
    preds = []
    