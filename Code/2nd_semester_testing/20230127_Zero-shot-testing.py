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
from Dataloaders import ReMapDataModule, ReMapDataModule_predict
from Models import FullTFModel

# Setup wandb logging
wandb.finish()
wandb.init(project="Thesis_experiments", entity="ntourne")
wandb_logger = WandbLogger(name='Small_experiment',project='pytorchlightning')
wandb_logger.experiment.config["Model"] = "Full"

# Data location
f = h5torch.File("/home/natant/Thesis/Data/ReMap_Processed/remap_all.h5t")

# Create indices to samples
indices = np.array([0] + list(np.cumsum(f["unstructured/chrom_lens"][:] + 10000)))
start_pos_each_chrom = indices[:-1]
end_pos_each_chrom = indices[1:] - 10000

sample_window_size = 1024
resolution = 128
positions_to_sample = []
for starts, stops in zip(start_pos_each_chrom, end_pos_each_chrom):
    positions_to_sample.append(np.arange(starts, stops, resolution)[:-10])
positions_to_sample = np.concatenate(positions_to_sample)

# TF that should be used in the model
TF_list = ['ZNF143', 'ZNF274', 'ZNF24', 'ZNF18']
TF_predict_list =  ['ZNF398', 'ZNF207', 'ZNF18']

# Create datamodule
remap_datamodule = ReMapDataModule(
    "/home/natant/Thesis/Data/ReMap_Processed/remap_all.h5t",
    TF_list=TF_list,
    positions_to_sample=positions_to_sample,
    window_size=sample_window_size,
    resoltution_factor=resolution
    ) 
remap_datamodule_predict = ReMapDataModule_predict(
    "/home/natant/Thesis/Data/ReMap_Processed/remap_all.h5t",
    TF_list=TF_predict_list,
    positions_to_sample=positions_to_sample,
    window_size=sample_window_size,
    resoltution_factor=resolution
    ) 

# Create model
Full_model = FullTFModel(   
    seq_len=sample_window_size,
    prot_embedding_dim=320,
    num_classes=len(TF_list),
    num_DNA_filters=20,
    num_prot_filters=20,
    DNA_kernel_size=10,
    prot_kernel_size=10,
    AdaptiveMaxPoolingOutput=600,
    dropout=0.25,
    num_linear_layers=3,
    linear_layer_size=64,
    final_embeddings_size=64
    )

# Create unique date timestamp
date = datetime.now().strftime("%Y%m%d_%H:%M:%S")

# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='/home/natant/Thesis/Data/Model_checkpoints',
    filename='Full-model-'+date+'-{epoch:02d}-{val_loss:.2f}'
    )


trainer = pl.Trainer(
    max_epochs = 5, 
    accelerator = "gpu", 
    devices = 1, 
    limit_train_batches = 100,
    limit_val_batches = 20,
    limit_predict_batches = 200,
    limit_test_batches = 200,
    callbacks=[checkpoint_callback],
    logger = wandb_logger
    )
trainer.fit(Full_model, datamodule=remap_datamodule)
trainer.test(Full_model, datamodule=remap_datamodule)
trainer.save_checkpoint('/home/natant/Thesis/Data/Model_checkpoints/Full-model-'+date+'.ckpt')

preds = trainer.predict(Full_model, datamodule=remap_datamodule_predict)
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

