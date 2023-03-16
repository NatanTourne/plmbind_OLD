import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import h5torch
import torch
import numpy as np
import pytorch_lightning as pl
import pickle
from torchmetrics.classification import MultilabelAUROC
# Own imports
from plmbind.data import ReMapDataModule
from plmbind.models import FullTFModel


def get_mean_embeddings(h5file, emb_name, TF_list):
    f = h5torch.File(h5file)
    protein_index_list = np.concatenate([np.where(f["unstructured/protein_map"][:].astype('str') == i) for i in TF_list]).flatten()
    embedding_list = []
    for i in protein_index_list:
        embedding_list.append(f[emb_name][str(i)][:])
    f.close()
    return torch.tensor(np.array(embedding_list)).mean(dim=1)

def get_latent_vectors(model, h5file, emb_name, TF_list):
    f = h5torch.File(h5file)
    protein_index_list = np.concatenate([np.where(f["unstructured/protein_map"][:].astype('str') == i) for i in TF_list]).flatten()
    embedding_list = []
    for i in protein_index_list:
        embedding_list.append(f[emb_name][str(i)][:])
    f.close()
    return model.get_TF_latent_vector(torch.tensor(np.array(embedding_list)))

def plot_auroc_barplot(df, x, y, num_plots, per_plot, color_list, save_loc = None, **plot_kwargs):
    fig, ax = plt.subplots(num_plots,1,figsize=(10,2.5*num_plots))
    for i in range(num_plots):
        sns.barplot(ax=ax[i], data = df[per_plot*i:per_plot*(i+1)], x = x, y = y, palette=color_list[per_plot*i:per_plot*(i+1)])
        ax[i].set_ylim(0,1)
        ax[i].axhline(0.5, ls='--')
        ax[i].tick_params(axis='x', rotation=90)
    plt.tight_layout()
    if save_loc:
        plt.savefig(save_loc)
    return fig, ax

# settings
model_loc = "/home/natant/Thesis-plmbind/Results/20230314/Full-model-20230313_09:33:07-epoch=15-val_loss=0.03.ckpt"
output_loc = "/home/natant/Thesis-plmbind/Results/20230314/"

model = FullTFModel.load_from_checkpoint(model_loc)

sample_window_size = 2**16 #32_768 #(2**15)
resolution = 128 # if you change this you also have to change your model definition


with open("/home/natant/Thesis-plmbind/Thesis/utils/TF_split/train_TFs", "rb") as f: 
    train_TFs = pickle.load(f) #[:50]
with open("/home/natant/Thesis-plmbind/Thesis/utils/TF_split/val_TFs", "rb") as f:
    val_TFs = pickle.load(f)

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
    embeddings="unstructured/t6_320_pad_trun",
    batch_size=16
    ) 

trainer = pl.Trainer(
    max_epochs = 10000, 
    accelerator = "gpu",
    devices = 1
    )



# Processing train TFs
train_TF_splits = [train_TFs[0:100], train_TFs[100:200], train_TFs[200:300], train_TFs[300:400], train_TFs[400:500], train_TFs[500:600], train_TFs[600:]]
pred_AUROC_train_TFs = []
for used_TFs in train_TF_splits:
    remap_datamodule.predict_setup(used_TFs, "val")
    preds = trainer.predict(model, datamodule=remap_datamodule)
    AUROC_train_TFs = MultilabelAUROC(num_labels=len(used_TFs), average='none')
    y_tensor = preds[0][0]
    y_hat_tensor = preds[0][1]
    for i in range(1,len(preds)):
        y_tensor = torch.concat([y_tensor, preds[i][0]])
        y_hat_tensor = torch.concat([y_hat_tensor, preds[i][1]])
    pred_AUROC_train_TFs.extend(AUROC_train_TFs(y_hat_tensor, y_tensor).numpy())
    print(len(pred_AUROC_train_TFs))



train_latent_vectors = get_latent_vectors(
    model=model, 
    h5file = "/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/train_no_alts.h5t", 
    emb_name="unstructured/t6_320_pad_trun", 
    TF_list = train_TFs
    )

train_mean_embs = get_mean_embeddings(
    h5file = "/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/train_no_alts.h5t", 
    emb_name="unstructured/t6_320_pad_trun", 
    TF_list = train_TFs
    )

train_TFs_val_DNA = pd.DataFrame({
    "HGNC symbol": train_TFs, 
    "AUROC": pred_AUROC_train_TFs,
    "latent_vecs": [train_latent_vectors[i].detach() for i in range(train_latent_vectors.shape[0])],
    "mean_emb": [train_mean_embs[i].detach() for i in range(train_mean_embs.shape[0])]})

info_df = pd.read_csv(r"/home/natant/Thesis-plmbind/Testing_ground/info.csv")
info_df_train = info_df[[i in train_TFs for i in info_df["HGNC symbol"]]]

result_df_train = info_df_train.join(train_TFs_val_DNA.set_index('HGNC symbol'), on='HGNC symbol')#.sort_values('AUROC', ascending=False)

color_dict = {"C2H2 ZF": "#c14454", "Homeodomain": "#f8a270", "bHLH": "#f27329", "other": "#4e84bc"}
color_map = [color_dict[i] if i in color_dict.keys() else color_dict["other"] for i in result_df_train["DBD"]]
fig, ax = plot_auroc_barplot(result_df_train, x = "HGNC symbol", y = "AUROC",num_plots = 9, per_plot=72, color_list = color_map, save_loc = output_loc+"train_TFs_val_DNA.png")

result_df_train.to_csv(output_loc+"train_TFs_val_DNA.csv")

