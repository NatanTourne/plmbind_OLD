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
from plmbind.data import ReMapDataModule_contrastive
from plmbind.models import PlmbindContrastive


# arg parser
parser = argparse.ArgumentParser(
    prog='Train_model',
    description='This trains the full two branch model.')

parser.add_argument("--out_dir", help="Dictory for output files", required=True)
parser.add_argument("--prot_branch", help="small or big", default="small")
parser.add_argument("--early_stop", help="Early stopping parameter. Either val_loss_DNA or val_loss_TF", default="val_loss_DNA")
parser.add_argument("--emb", help="Protein embeddings to use", default = "t6_320_pad_trun")
parser.add_argument("--emb_dim", help="The size of the embeddings", type=int, default=320)
parser.add_argument("--batch_size", help="The batch size", type=int, default=256)
parser.add_argument("--TF_batch_size",  help="The number of TFs to subsample", type=int, default=0)

parser.add_argument("--num_DNA_filters", type=int, default=50)
parser.add_argument("--num_prot_filters", type=int, default=50)
parser.add_argument("--DNA_kernel_size", type=int, default=10)
parser.add_argument("--prot_kernel_size", type=int, default=10)
parser.add_argument("--prot_dropout", type=float, default=0.4)
parser.add_argument("--DNA_dropout", type=float, default=0.25)
parser.add_argument("--latent_vector_size", type=int, default=64)
parser.add_argument("--calculate_val_TF_loss", type=bool, default=True)
parser.add_argument("--learning_rate_DNA_branch", type=float, default=0.00001)
parser.add_argument("--learning_rate_protein_branch", type=float, default=0.00001)
parser.add_argument("--loss_function", default="BCE")
parser.add_argument("--loss_weights", default=None, nargs="+", type=float)

parser.add_argument("--max_epochs", type=int, default=1000)
parser.add_argument("--limit_train_batches", type=float, default=1.0)
parser.add_argument("--limit_val_batches", type=float, default=1.0)

parser.add_argument("--pre_trained_DNA_branch", default = "None")

parser.add_argument("--Train_TFs_loc",help="Location of pickled list of train TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/train_TFs")
parser.add_argument("--Val_TFs_loc", help="Location of pickled list of validation TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/val_TFs")
parser.add_argument("--Test_TFs_loc", help="Location of pickled list of test TFs", default = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/test_TFs")
parser.add_argument("--train_loc", help="Location of training data",default ="/home/data/shared/natant/Data/contrastive_train_no_alts.h5t")
parser.add_argument("--val_loc", help="Location of validation data", default ="/home/data/shared/natant/Data/contrastive_val_no_alts.h5t")
parser.add_argument("--test_loc", help="Location of test data", default ="/home/data/shared/natant/Data/contrastive_test_no_alts.h5t")

args = parser.parse_args()

# Create unique date timestamp
date = datetime.now().strftime("%Y%m%d_%H:%M:%S")

if args.out_dir[-1] != "/":
    args.out_dir = args.out_dir+"/"

settings = vars(args)
settings["date"] = date
settings["loss_weights"] = [settings["loss_weights"]]
settings_df = pd.DataFrame(settings, index=[0])
settings_df.to_csv(args.out_dir+"settings.csv")


Embeddings = "unstructured/" + args.emb

# Setup wandb logging
wandb.finish()
wandb.init(project="Thesis_experiments", entity="ntourne")
wandb_logger = WandbLogger(name='Small_experiment',project='pytorchlightning')
wandb_logger.experiment.config["Model"] = "two_branch"
wandb_logger.experiment.config["Embeddings"] = args.emb

# Load list of TFs used for training (embeddings will be fetched from dataloader)
with open(args.Train_TFs_loc, "rb") as f: 
    train_TFs = pickle.load(f)
with open(args.Val_TFs_loc, "rb") as f:
    val_TFs = pickle.load(f)

# Create datamodule:
    # Seperate files for train, val, test
    # Protein embeddings are now specified (multiple sizes are possible)
    

remap_datamodule = ReMapDataModule_contrastive(
    train_loc=args.train_loc,
    val_loc=args.val_loc,
    test_loc=args.test_loc,
    train_TFs=train_TFs,
    val_TFs=val_TFs,
    TF_batch_size=args.TF_batch_size, # PUT 0 WHEN YOU WANT TO USE ALL TFs
    embeddings=Embeddings,
    batch_size=args.batch_size#2048
    )

      
print(args.loss_weights)
Full_model = PlmbindContrastive( 
        prot_embedding_dim=args.emb_dim,
        num_DNA_filters=args.num_DNA_filters,
        num_prot_filters=args.num_prot_filters,
        DNA_kernel_size=args.DNA_kernel_size,
        prot_kernel_size=args.prot_kernel_size,
        DNA_dropout=args.DNA_dropout,
        protein_dropout=args.prot_dropout,
        final_embeddings_size=args.latent_vector_size,
        calculate_val_tf_loss=args.calculate_val_TF_loss,
        learning_rate_protein_branch=args.learning_rate_protein_branch,
        learning_rate_DNA_branch=args.learning_rate_DNA_branch,
        DNA_branch_path = args.pre_trained_DNA_branch,
        loss_function=args.loss_function,
        loss_weights = args.loss_weights[0]
        )


# Create checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss_DNA',
    dirpath=args.out_dir,
    filename='Full-model-train_TF_loss-'+date+'-{epoch:02d}-{val_loss_DNA:.2f}-{val_loss_TF:.2f}'
    )
checkpoint_callback_val_TF = ModelCheckpoint(
    monitor='val_loss_TF',
    dirpath=args.out_dir,
    filename='Contrastive-model-val_TF_loss-'+date+'-{epoch:02d}-{val_loss_DNA:.2f}-{val_loss_TF:.2f}'
    )

# Create early stopping callback
early_stopping = EarlyStopping(args.early_stop)

# Create Trainer
trainer = pl.Trainer(
    max_epochs = args.max_epochs,
    limit_train_batches=args.limit_train_batches,
    limit_val_batches=args.limit_val_batches,
    accelerator = "gpu", 
    devices = [2],
    callbacks=[checkpoint_callback, checkpoint_callback_val_TF, early_stopping],
    logger = wandb_logger
    )

# Create 

# Fit model
trainer.fit(Full_model, datamodule=remap_datamodule)

# Save checkpoint
trainer.save_checkpoint(args.out_dir + 'Contrastive-model-'+date+'.ckpt')