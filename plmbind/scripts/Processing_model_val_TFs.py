####
# This will later be used to test the test the test TFs on test DNA
####
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

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

## settings
model_loc = "/home/natant/Thesis-plmbind/Results/20230317/Full-model-DNA-20230319_09:21:19-epoch=10-val_loss=0.00.ckpt"
output_loc = "/home/natant/Thesis-plmbind/Results/20230317/train_TF_model_epoch10_"
embeddings = "unstructured/t6_320_pad_trun"

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
    val_list=val_TFs,
    TF_batch_size=0, # PUT 0 WHEN YOU WANT TO USE ALL TFs
    window_size=sample_window_size,
    resolution_factor=resolution,
    embeddings=embeddings,
    batch_size=16
    ) 

trainer = pl.Trainer(
    max_epochs = 10000, 
    accelerator = "gpu", 
    devices = 1
    )


# Processing Validation TFs
remap_datamodule.predict_setup(val_TFs, "val")
preds = trainer.predict(model, datamodule=remap_datamodule)


AUROC_val_TFs = MultilabelAUROC(num_labels=len(val_TFs), average='none')
targets_list = []
pred_list = []
for X in preds:
    targets_list.append(X[0])
    pred_list.append(X[1])
pred_AUROC_val_TFs = AUROC_val_TFs(torch.cat(pred_list), torch.cat(targets_list))

train_latent_vectors = get_latent_vectors(
    model=model, 
    h5file = "/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/train_no_alts.h5t", 
    emb_name=embeddings, 
    TF_list = train_TFs
    )
val_latent_vectors = get_latent_vectors(
    model=model, 
    h5file = "/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/train_no_alts.h5t", 
    emb_name=embeddings, 
    TF_list = val_TFs
    )
with open(output_loc+'train_latent_vectors.pkl', 'wb') as f:
    pickle.dump(train_latent_vectors, f)
with open(output_loc+'val_latent_vectors.pkl', 'wb') as f:
    pickle.dump(val_latent_vectors, f)
cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

highest_cos = [float(cos(test_vector, train_latent_vectors).detach().max()) for test_vector in val_latent_vectors]

Most_similar = [train_TFs[int(torch.argmax(cos(test_vector, train_latent_vectors)))] for test_vector in val_latent_vectors]

train_mean_embs = get_mean_embeddings(
    h5file = "/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/train_no_alts.h5t", 
    emb_name=embeddings, 
    TF_list = train_TFs
    )
val_mean_embs = get_mean_embeddings(
    h5file = "/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/train_no_alts.h5t", 
    emb_name=embeddings, 
    TF_list = val_TFs
    )

with open(output_loc + 'train_mean_embs.pkl', 'wb') as f:
    pickle.dump(train_mean_embs, f)
with open(output_loc + 'val_mean_embs.pkl', 'wb') as f:
    pickle.dump(val_mean_embs, f)
highest_cos_emb = [float(cos(test_vector, train_mean_embs).detach().max()) for test_vector in val_mean_embs]   
Most_similar_emb = [train_TFs[int(torch.argmax(cos(test_vector, train_mean_embs)))] for test_vector in val_mean_embs]


zero_shot_TFs = pd.DataFrame({
    "HGNC symbol": val_TFs, 
    "AUROC": pred_AUROC_val_TFs,
    "latent_vecs": [val_latent_vectors[i].detach() for i in range(val_latent_vectors.shape[0])],
    "max_latent_cos": highest_cos, 
    "most_similar_latent": Most_similar,
    "mean_emb": [val_mean_embs[i].detach() for i in range(val_mean_embs.shape[0])],
    "highest_cos_emb": highest_cos_emb,
    "Most_similar_emb": Most_similar_emb})

info_df = pd.read_csv(r"/home/natant/Thesis-plmbind/Testing_ground/info.csv")
info_df_val = info_df[[i in val_TFs for i in info_df["HGNC symbol"]]]

result_df_val = info_df_val.join(zero_shot_TFs.set_index('HGNC symbol'), on='HGNC symbol')#.sort_values('AUROC', ascending=False)

color_dict = {"C2H2 ZF": "#c14454", "Homeodomain":"#f8a270", "bHLH": "#f27329", "other": "#4e84bc"}
color_map = [color_dict[i] if i in color_dict.keys() else color_dict["other"] for i in result_df_val["DBD"]]
fig, ax = plot_auroc_barplot(result_df_val, x = "HGNC symbol", y = "AUROC",num_plots = 2, per_plot=52, color_list = color_map, save_loc = output_loc + "val_TFs_val_DNA.png")

result_df_val.to_csv(output_loc + "val_TFs_val_DNA.csv")

