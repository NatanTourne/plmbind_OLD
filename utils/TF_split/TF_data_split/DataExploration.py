## SHOULD BE RUN ON SERVER, JUST PUT IT HERE TO KNOW WHERE THE PICKLED FILES CAME FROM!!!
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def get_chrom_number(chr):
    try:
        num = int(chr[3:])
    except:
        num = 100
    return num

TF_list = []
for filename in os.scandir("/home/natant/Thesis/Data/ReMap/TF_bed_files_fixed"):
    if filename.name[-4:] == ".bed":
        TF_list.append(filename.name[:-4])
TF_list

count = 0
All_chrom_names = []
for TF_name in TF_list:
    count += 1
    if count > 10000:
        break
    bed_loc = "/home/natant/Thesis/Data/ReMap/TF_bed_files_fixed/" + TF_name +  ".bed"
    TF_DF = pd.read_csv(bed_loc, sep='\t', header = None, usecols=[0, 1, 2])
    TF_DF.columns = ["chr", "chromStart", "chromEnd"]
    chrom_numbers = TF_DF["chr"].value_counts()
    All_chrom_names.append(chrom_numbers.index)

Unique_chrom_names = np.unique(np.concatenate(All_chrom_names))
chrom_order = sorted(list(Unique_chrom_names), key=lambda x: get_chrom_number(x))
all_TF_DF = pd.DataFrame(chrom_order)
all_TF_DF.columns=["chrom_names"]
all_TF_DF = all_TF_DF.set_index("chrom_names")

count = 0
for TF_name in TF_list:
    count += 1
    if count > 10000:
        break
    bed_loc = "/home/natant/Thesis/Data/ReMap/TF_bed_files_fixed/" + TF_name +  ".bed"
    TF_DF = pd.read_csv(bed_loc, sep='\t', header = None, usecols=[0, 1, 2, 3])
    TF_DF.columns = ["chr", "chromStart", "chromEnd", "Celltype"]
    chrom_numbers = TF_DF["chr"].value_counts()
    all_TF_DF[TF_name] = chrom_numbers

all_TF_DF.to_pickle("/home/natant/Thesis/Code/Testing/Data_exploration/all_TF_DF.pkl")


cellType_list = []
count = 0
for TF_name in TF_list:
    count += 1
    if count > 10000:
        break
    bed_loc = "/home/natant/Thesis/Data/ReMap/TF_bed_files_fixed/" + TF_name +  ".bed"
    TF_DF = pd.read_csv(bed_loc, sep='\t', header = None, usecols=[0, 1, 2, 3])
    TF_DF.columns = ["chr", "chromStart", "chromEnd", "Celltype"]
    cellType_list.append(pd.DataFrame({TF_name: pd.DataFrame(np.concatenate(TF_DF["Celltype"].apply(lambda x: x.split(":")[1].split(",")))).value_counts()}))

all_celltypes = pd.concat(cellType_list, axis=1)
all_celltypes.to_pickle("/home/natant/Thesis/Code/Testing/Data_exploration/all_celltypes.pkl")



