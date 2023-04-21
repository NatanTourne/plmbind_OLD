import numpy as np
from pandas import date_range
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
import pickle

# Own imports
from plmbind.data import ReMapDataModule2_0
from plmbind.models import PlmbindFullModel

from pytorch_lightning.callbacks import EarlyStopping

# Setup wandb logging
wandb.finish()
wandb.init(project="Thesis_experiments", entity="ntourne")
wandb_logger = WandbLogger(name='Small_experiment',project='pytorchlightning')
wandb_logger.experiment.config["Model"] = "small_double_val"
wandb_logger.experiment.config["Embeddings"] = "unstructured/t6_320_pad_trun"
wandb_logger.experiment.config["Resolution"] = 128

# sample window and resolution
sample_window_size = 2**16 #32_768 #(2**15)
resolution = 128 # if you change this you also have to change your model definition

# Load list of TFs used for training (embeddings will be fetched from dataloader)
with open("/home/natant/Thesis-plmbind/Thesis/utils/TF_split/train_TFs", "rb") as f: 
    train_TFs = pickle.load(f)
with open("/home/natant/Thesis-plmbind/Thesis/utils/TF_split/val_TFs", "rb") as f:
    val_TFs = pickle.load(f)

# Create datamodule:
    # Seperate files for train, val, test
    # Protein embeddings are now specified (multiple sizes are possible)
remap_datamodule = ReMapDataModule2_0(
    train_loc="/home/data/shared/natant/Data/train_no_alts.h5t",
    val_loc="/home/data/shared/natant/Data/val_no_alts.h5t",
    test_loc="/home/data/shared/natant/Data/test_no_alts.h5t",
    TF_list=train_TFs,
    val_list=val_TFs,
    TF_batch_size=0, # PUT 0 WHEN YOU WANT TO USE ALL TFs
    window_size=sample_window_size,
    embeddings="unstructured/t6_320_pad_trun",
    batch_size=16
    ) 

# Create model
Full_model = PlmbindFullModel(   
    seq_len=sample_window_size,
    prot_embedding_dim=320,
    num_DNA_filters=50,
    num_prot_filters=25,
    DNA_kernel_size=10,
    prot_kernel_size=5,
    final_embeddings_size=128,
    calculate_val_tf_loss=True,
    DNA_branch_path = "/home/natant/Thesis-plmbind/Testing_ground/testing_DL2_0/test_module/DNA-model-20230413_14:47:06-epoch=00-val_loss=0.69.ckpt"
    )

# Create unique date timestamp
date = datetime.now().strftime("%Y%m%d_%H:%M:%S")
# Set output directory
out_dir = "/home/natant/Thesis-plmbind/Results/20230322_2/" # Do not forget the last '/' !! 
# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss_DNA',
    dirpath=out_dir,
    filename='Full-model-DNA-'+date+'-{epoch:02d}-{val_loss:.2f}'
    )
checkpoint_callback_val_TF = ModelCheckpoint(
    monitor='val_loss_TF',
    dirpath=out_dir,
    filename='Full-model-TF-'+date+'-{epoch:02d}-{val_loss:.2f}'
    )

# Create early stopping callback
early_stopping = EarlyStopping('val_loss_DNA')

# Create Trainer
trainer = pl.Trainer(
    max_epochs = 10000, 
    accelerator = "gpu", 
    devices = 1,
    callbacks=[checkpoint_callback, checkpoint_callback_val_TF, early_stopping],
    logger = wandb_logger
    )

# Fit model
trainer.fit(Full_model, datamodule=remap_datamodule)

# Save checkpoint
trainer.save_checkpoint(out_dir + 'Full-model-'+date+'.ckpt')