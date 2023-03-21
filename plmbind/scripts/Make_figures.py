import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle



save_loc = "/home/natant/Thesis-plmbind/Results/20230316/"

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

with open("/home/natant/Thesis-plmbind/Thesis/utils/TF_split/train_TFs", "rb") as f: 
    train_TFs = pickle.load(f) #[:50]
with open("/home/natant/Thesis-plmbind/Thesis/utils/TF_split/val_TFs", "rb") as f:
    val_TFs = pickle.load(f)
    
val_TF_val_DNA = pd.read_csv(save_loc + "val_TFs_val_DNA.csv")
val_TF_val_DNA = val_TF_val_DNA.set_index("HGNC symbol", drop=False).loc[val_TFs]
train_TF_val_DNA = pd.read_csv(save_loc + "train_TFs_val_DNA.csv")
train_TF_val_DNA = train_TF_val_DNA.set_index("HGNC symbol", drop=False).loc[train_TFs]

Sorted_val_TF_val_DNA = val_TF_val_DNA.sort_values('AUROC', ascending=False)
color_dict = {"C2H2 ZF": "#c14454", "Homeodomain":"#f8a270", "bHLH": "#f27329", "other": "#4e84bc"}
color_map = [color_dict[i] if i in color_dict.keys() else color_dict["other"] for i in Sorted_val_TF_val_DNA["DBD"]]
fig, ax = plot_auroc_barplot(Sorted_val_TF_val_DNA, x = "HGNC symbol", y = "AUROC",num_plots = 2, per_plot=52, color_list = color_map, save_loc = save_loc+"val_TFs_val_DNA_barplot.png")

fig, ax = plt.subplots(figsize=(8,5))
color_dict = {"C2H2 ZF": "#c14454", "Homeodomain":"#f8a270", "bHLH": "#f27329", "other": "#4e84bc"}
color_map = [color_dict[i] if i in color_dict.keys() else color_dict["other"] for i in val_TF_val_DNA["DBD"].unique().tolist()]
sns.boxplot(ax=ax, data=val_TF_val_DNA, x="DBD", y="AUROC", palette=color_map)
sns.stripplot(ax=ax, data=val_TF_val_DNA, x="DBD", y="AUROC", size=5, color="lightgray", linewidth=0)
ax.set_ylim(0,1)
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig(save_loc+"val_TFs_val_DNA_boxplot.png")

Sorted_train_TF_val_DNA = train_TF_val_DNA.sort_values('AUROC', ascending=False)
color_dict = {"C2H2 ZF": "#c14454", "Homeodomain":"#f8a270", "bHLH": "#f27329", "other": "#4e84bc"}
color_map = [color_dict[i] if i in color_dict.keys() else color_dict["other"] for i in Sorted_train_TF_val_DNA["DBD"]]
fig, ax = plot_auroc_barplot(Sorted_train_TF_val_DNA, x = "HGNC symbol", y = "AUROC",num_plots = 9, per_plot=72, color_list = color_map, save_loc = save_loc+"train_TFs_val_DNA_barplot.png")

order = train_TF_val_DNA.groupby("DBD").median().sort_values("AUROC", ascending=False).index.to_list()
fig, ax = plt.subplots(figsize=(15,5))
color_dict = {"C2H2 ZF": "#c14454", "Homeodomain":"#f8a270", "bHLH": "#f27329", "other": "#4e84bc"}
color_map = [color_dict[i] if i in color_dict.keys() else color_dict["other"] for i in order] # train_TF_val_DNA["DBD"].unique().tolist()]
sns.boxplot(ax=ax, data=train_TF_val_DNA, x="DBD", y="AUROC", palette=color_map, order=order)
sns.stripplot(ax=ax, data=train_TF_val_DNA, x="DBD", y="AUROC", order=order, size=5, color="lightgray", linewidth=0)
ax.set_ylim(0,1)
plt.xticks(rotation = 90)
plt.tight_layout()
plt.savefig(save_loc+"train_TFs_val_DNA_boxplot.png")

Sorted_val_TF_val_DNA["split"] = ["val"]*len(Sorted_val_TF_val_DNA)
Sorted_train_TF_val_DNA["split"] = ["train"]*len(Sorted_train_TF_val_DNA)
all_results = pd.concat([Sorted_train_TF_val_DNA, Sorted_val_TF_val_DNA])
all_results["DBD_reduced"] = [i if i in ["C2H2 ZF", "Homeodomain", "bHLH"] else "other" for i in all_results["DBD"].to_list()] 
boxplot_val_train = sns.boxplot(data=all_results, x = "DBD_reduced", y = "AUROC", hue = "split")
boxplot_val_train.set(ylim=(0, 1))
boxplot_val_train.figure.savefig(save_loc+"DBD_train_val_boxplot.png")

