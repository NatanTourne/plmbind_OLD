import numpy as np
from pandas import date_range
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
import pickle

# Own imports
from plmbind.data import ReMapDataModule
from plmbind.models import FullTFModel

from pytorch_lightning.callbacks import EarlyStopping

# Setup wandb logging
wandb.finish()
wandb.init(project="Thesis_experiments", entity="ntourne")
wandb_logger = WandbLogger(name='Small_experiment',project='pytorchlightning')
wandb_logger.experiment.config["Model"] = "Full"
wandb_logger.experiment.config["Embeddings"] = "unstructured/t12_480_pad_trun"
wandb_logger.experiment.config["Resolution"] = 128

# sample window and resolution
sample_window_size = 2**16 #32_768 #(2**15)
resolution = 128 # if you change this you also have to change your model definition

# Load list of TFs used for training (embeddings will be fetched from dataloader)
with open("/home/natant/Thesis-plmbind/Thesis/utils/TF_split/train_TFs", "rb") as f: 
    train_TFs = pickle.load(f)

# Create datamodule:
    # Seperate files for train, val, test
    # Protein embeddings are now specified (multiple sizes are possible)
remap_datamodule = ReMapDataModule(
    train_loc="/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/train_no_alts.h5t",
    val_loc="/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/val_no_alts.h5t",
    test_loc="/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/test_no_alts.h5t",
    TF_list=train_TFs,
    TF_batch_size=0, # PUT 0 WHEN YOU WANT TO USE ALL TFs
    window_size=sample_window_size,
    resolution_factor=resolution,
    embeddings="unstructured/t12_480_pad_trun",         #"unstructured/t6_320_pad_trun",
    batch_size=8
    ) 

# Create model
Full_model = FullTFModel(   
    seq_len=sample_window_size,
    prot_embedding_dim=480,
    TF_list=train_TFs,
    num_classes=len(train_TFs), # SHOULD BE THE SAME AS THE TF_BATCH_SIZE
    num_DNA_filters=50,
    num_prot_filters=50,
    DNA_kernel_size=10,
    prot_kernel_size=10,
    dropout=0.25,
    final_embeddings_size=128
    )

# Create unique date timestamp
date = datetime.now().strftime("%Y%m%d_%H:%M:%S")
# Set output directory
out_dir = "/home/natant/Thesis-plmbind/Results/20230316/"
# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=out_dir,
    filename='Full-model-'+date+'-{epoch:02d}-{val_loss:.2f}'
    )

# Create early stopping callback
early_stopping = EarlyStopping('val_loss')

# Create Trainer
trainer = pl.Trainer(
    max_epochs = 10000, 
    accelerator = "gpu", 
    devices = 1,
    callbacks=[checkpoint_callback, early_stopping],
    logger = wandb_logger
    )

# Fit model
trainer.fit(Full_model, datamodule=remap_datamodule)

# Save checkpoint
trainer.save_checkpoint(out_dir + 'Full-model-'+date+'.ckpt')