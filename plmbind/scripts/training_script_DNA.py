import numpy as np
import h5torch
import numpy as np
from pandas import date_range
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime

# Own imports
from plmbind.data import ReMapDataModule
from plmbind.models import MultilabelModel

# Setup wandb logging
wandb.finish()
wandb.init(project="Thesis_experiments", entity="ntourne")
wandb_logger = WandbLogger(name='Small_experiment',project='pytorchlightning')
wandb_logger.experiment.config["Model"] = "DNA"

sample_window_size = 1024
resolution = 128

# TFs included in the training dataset
TF_list = ['ZNF143', 'ZNF274', 'ZNF24', 'ZNF18']

# Create datamodule:
    # Seperate files for train, val, test
    # Protein embeddings are now specified (multiple sizes are possible)
remap_datamodule = ReMapDataModule(
    train_loc="/home/natant/Thesis/Data/ReMap_TRAIN_TEST_SPLIT/remap_train.h5t",
    val_loc="/home/natant/Thesis/Data/ReMap_TRAIN_TEST_SPLIT/remap_val_short.h5t",
    test_loc="/home/natant/Thesis/Data/ReMap_TRAIN_TEST_SPLIT/remap_test_short.h5t",
    TF_list=TF_list,
    window_size=sample_window_size,
    resolution_factor=resolution,
    embeddings="unstructured/prot_embeddings_t6"
    ) 


# Create model
DNA_model = MultilabelModel(   
    seq_len=sample_window_size,
    num_classes=len(TF_list),
    num_DNA_filters=50,
    DNA_kernel_size=10,
    dropout=0.25,
    num_linear_layers=3,
    linear_layer_size=128
    )

# Create unique date timestamp
date = datetime.now().strftime("%Y%m%d_%H:%M:%S")

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='/home/natant/Thesis/Logs/Model_checkpoints',
    filename='DNA-model-'+date+'-{epoch:02d}-{val_loss:.2f}'
    )


trainer = pl.Trainer(
    max_epochs = 5, 
    accelerator = "gpu", 
    devices = 1, 
    limit_train_batches = 10,
    limit_val_batches = 20,
    limit_predict_batches = 20,
    limit_test_batches = 100,
    callbacks=[checkpoint_callback],
    logger = wandb_logger
    )
trainer.fit(DNA_model, datamodule=remap_datamodule)
trainer.test(DNA_model, datamodule=remap_datamodule)
trainer.save_checkpoint('/home/natant/Thesis/Logs/Model_checkpoints/DNA-model-'+date+'.ckpt')



## Well the model can not do One-shot prediction so this is just the same as trainer.test!!?
remap_datamodule.predict_setup(TF_list, "test")
preds = trainer.predict(DNA_model, datamodule=remap_datamodule)
from torchmetrics.classification import MultilabelAUROC
AUROC_test = MultilabelAUROC(num_labels=len(TF_list), average='none')
targets_list = []
pred_list = []
for X in preds:
    targets_list.append(X[0])
    pred_list.append(X[1])
import torch
target_tensor = torch.cat(targets_list)
pred_tensor = torch.cat(pred_list)
pred_AUROC = AUROC_test(pred_tensor, target_tensor)
print(pred_AUROC)