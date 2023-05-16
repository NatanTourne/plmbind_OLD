# imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
from datetime import datetime
import pickle
from pytorch_lightning.callbacks import EarlyStopping
import argparse
import pandas as pd

# Own imports
from plmbind.data import ReMapDataModule2_0
from plmbind.models import MultilabelModel


# arg parser
parser = argparse.ArgumentParser(
    prog='Train_model',
    description='This trains the DNA branch model.')

parser.add_argument("--out_dir", help="Dictory for output files", required=True)
parser.add_argument("--window_size", help="The window size", type=int, default=2**16)
parser.add_argument("--batch_size", help="The batch size", type=int, default=16)
parser.add_argument("--emb", help="Protein embeddings to use", default = "t6_320_pad_trun")
parser.add_argument("--num_DNA_filters", type=int, default=50)
parser.add_argument("--DNA_kernel_size", type=int, default=10)
parser.add_argument("--DNA_dropout", type=float, default=0.25)
parser.add_argument("--learning_rate", type=float, default=0.00001)
parser.add_argument("--max_epochs", type=int, default=1000)
parser.add_argument("--limit_train_batches", type=float, default=1.0)
parser.add_argument("--limit_val_batches", type=float, default=1.0)

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

settings = vars(args)
settings["date"] = date
settings_df = pd.DataFrame(settings, index=[0])
settings_df.to_csv(args.out_dir+"settings.csv")


Embeddings = "unstructured/" + args.emb

# Setup wandb logging
wandb.finish()
wandb.init(project="Thesis_experiments", entity="ntourne")
wandb_logger = WandbLogger(name='Small_experiment',project='pytorchlightning')
wandb_logger.experiment.config["Model"] = "DNA_model"

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

# Create model
Full_model = MultilabelModel(
        seq_len=args.window_size,
        num_classes=len(train_TFs),
        num_DNA_filters=args.num_DNA_filters,
        DNA_kernel_size=args.DNA_kernel_size,
        DNA_dropout=args.DNA_dropout,
        learning_rate=args.learning_rate
        )



# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=args.out_dir,
    filename='DNA-model-'+date+'-{epoch:02d}-{val_loss:.2f}'
    )

# Create early stopping callback
early_stopping = EarlyStopping("val_loss")

# Create Trainer
trainer = pl.Trainer(
    max_epochs = args.max_epochs, 
    limit_train_batches=args.limit_train_batches,
    limit_val_batches=args.limit_val_batches,
    accelerator = "gpu", 
    devices = [1],
    callbacks=[checkpoint_callback, early_stopping],
    logger = wandb_logger
    )

# Create 

# Fit model
trainer.fit(Full_model, datamodule=remap_datamodule)

# Save checkpoint
trainer.save_checkpoint(args.out_dir + 'DNA-model-'+date+'.ckpt')