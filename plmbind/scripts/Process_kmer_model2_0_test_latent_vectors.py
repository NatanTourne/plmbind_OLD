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
with open(args.Test_TFs_loc, "rb") as f:
    test_TFs = pickle.load(f)

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



def get_latent_vectors(model, h5file, emb_name, TF_list):
    f = h5torch.File(h5file)
    protein_index_list = np.concatenate([np.where(f["1/prots"][:].astype('str') == i) for i in TF_list]).flatten()
    embedding_list = []
    for i in protein_index_list:
        embedding_list.append(f[emb_name][str(i)][:])
    f.close()
    return model.get_TF_latent_vector(torch.tensor(np.array(embedding_list)))



##### val_TF_loss_model ####
Full_model = PlmbindFullModel_kmer.load_from_checkpoint(val_TF_loss_model_loc)


test_latent_vectors = get_latent_vectors(
    model=Full_model, 
    h5file = args.train_loc, 
    emb_name=Embeddings, 
    TF_list = test_TFs
    )

with open(args.out_dir+'val_TF_loss_model_test_latent_vectors.pkl', 'wb') as f:
    pickle.dump(test_latent_vectors, f)
    
train_latent_vectors = get_latent_vectors(
    model=Full_model, 
    h5file = args.train_loc, 
    emb_name=Embeddings, 
    TF_list = train_TFs
    )

with open(args.out_dir+'val_TF_loss_model_train_latent_vectors.pkl', 'wb') as f:
    pickle.dump(train_latent_vectors, f)

val_latent_vectors = get_latent_vectors(
    model=Full_model, 
    h5file = args.train_loc, 
    emb_name=Embeddings, 
    TF_list = val_TFs
    )

with open(args.out_dir+'val_TF_loss_model_val_latent_vectors.pkl', 'wb') as f:
    pickle.dump(val_latent_vectors, f)
    

Full_model = PlmbindFullModel_kmer.load_from_checkpoint(train_TF_loss_model_loc)

test_latent_vectors = get_latent_vectors(
    model=Full_model, 
    h5file = args.train_loc, 
    emb_name=Embeddings, 
    TF_list = test_TFs
    )

with open(args.out_dir+'train_TF_loss_model_test_latent_vectors.pkl', 'wb') as f:
    pickle.dump(test_latent_vectors, f)
    
train_latent_vectors = get_latent_vectors(
    model=Full_model, 
    h5file = args.train_loc, 
    emb_name=Embeddings, 
    TF_list = train_TFs
    )

with open(args.out_dir+'train_TF_loss_model_train_latent_vectors.pkl', 'wb') as f:
    pickle.dump(train_latent_vectors, f)
    
val_latent_vectors = get_latent_vectors(
    model=Full_model, 
    h5file = args.train_loc, 
    emb_name=Embeddings, 
    TF_list = val_TFs
    )

with open(args.out_dir+'train_TF_loss_model_val_latent_vectors.pkl', 'wb') as f:
    pickle.dump(val_latent_vectors, f)
