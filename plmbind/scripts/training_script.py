import numpy as np
import h5torch
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


# Setup wandb logging
wandb.finish()
wandb.init(project="Thesis_experiments", entity="ntourne")
wandb_logger = WandbLogger(name='Small_experiment',project='pytorchlightning')
wandb_logger.experiment.config["Model"] = "Full"

sample_window_size = 2048
resolution = 1024 #128

# TFs included in the training dataset
TF_list = ['ZNF143', 'ZNF274', 'ZNF24', 'ZNF18']
with open("/home/natant/Thesis/plmbind/data_processing/TF_split/ZNF_train", "rb") as f: 
    ZNF_train = pickle.load(f)
with open("/home/natant/Thesis/plmbind/data_processing/TF_split/ZNF_test", "rb") as f: 
    ZNF_test = pickle.load(f)
with open("/home/natant/Thesis/plmbind/data_processing/TF_split/ZNF_val", "rb") as f:
    ZNF_val = pickle.load(f)

# Create datamodule:
    # Seperate files for train, val, test
    # Protein embeddings are now specified (multiple sizes are possible)
remap_datamodule = ReMapDataModule(
    train_loc="/home/natant/Thesis/Data/ReMap2022/train.h5t",
    val_loc="/home/natant/Thesis/Data/ReMap2022/val.h5t",
    test_loc="/home/natant/Thesis/Data/ReMap2022/test.h5t",
    TF_list=ZNF_train[:50],
    TF_batch_size=0, # PUT 0 WHEN YOU WANT TO USE ALL TFs
    window_size=sample_window_size,
    resolution_factor=resolution,
    embeddings="unstructured/prot_embeddings_t6",
    batch_size=8 # HAS TO BE ONE WHEN, TF_BATCH_SIZE != 0
    ) 

# Create model


Full_model = FullTFModel(   
    seq_len=sample_window_size,
    prot_embedding_dim=320,
    TF_list=ZNF_train[:50],
    num_classes=len(ZNF_train[:50]), # SHOULD BE THE SAME AS THE TF_BATCH_SIZE
    num_DNA_filters=50,
    num_prot_filters=50,
    DNA_kernel_size=10,
    prot_kernel_size=10,
    AdaptiveMaxPoolingOutput=1000,
    dropout=0.25,
    num_linear_layers=5,
    linear_layer_size=128,
    final_embeddings_size=128
    )

# Create unique date timestamp
date = datetime.now().strftime("%Y%m%d_%H:%M:%S")

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='/home/natant/Thesis/Logs/Model_checkpoints',
    filename='Full-model-'+date+'-{epoch:02d}-{val_loss:.2f}'
    )


# Create Trainer
trainer = pl.Trainer(
    max_epochs = 10000, 
    accelerator = "gpu", 
    devices = 1, 
    limit_train_batches = 500,
    limit_val_batches = 50,
    limit_predict_batches = 200,
    limit_test_batches = 200,
    callbacks=[checkpoint_callback],
    logger = wandb_logger
    )

# Fit model
trainer.fit(Full_model, datamodule=remap_datamodule)

# Test Model
trainer.test(Full_model, datamodule=remap_datamodule)

# Save checkpoint
trainer.save_checkpoint('/home/natant/Thesis/Logs/Model_checkpoints/Full-model-'+date+'.ckpt')

### PREDICTION ###
# Specifiy the TFs for prediction (Include one of the training data as sanity check)
#TF_predict_list =  ZNF_test[:10]
TF_predict_list = ZNF_train[:50]
# Setup the datamodule for prediction: predict for "train", "test" or "val" datasplit
#remap_datamodule.predict_setup(TF_predict_list, "test")
remap_datamodule.predict_setup(TF_predict_list, "test")
# Do the prediction
preds = trainer.predict(Full_model, datamodule=remap_datamodule)

# This is still manual because of the issue with batch sizes, can now be simplified similar to the other metrics ##!! TODO
from torchmetrics.classification import MultilabelAUROC
AUROC_test = MultilabelAUROC(num_labels=len(TF_predict_list), average='none')
targets_list = []
pred_list = []

for X in preds:
    targets_list.append(X[0])
    pred_list.append(X[1])
import torch
target_tensor = torch.cat(targets_list)
pred_tensor = torch.cat(pred_list)
pred_AUROC = AUROC_test(pred_tensor, target_tensor)
print("---------------------------------------------------------")
print("ZERO-SHOT PREDICTION:")
print(pred_AUROC)
print("(as sanity check the last value is normal validation data)")

