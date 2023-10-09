import argparse
import pandas as pd
import numpy as np
import h5torch
import py2bit


bed_path = "/home/data/shared/natant/Data/remap2022_nr_macs2_hg38_v1_0.bed" 
genome_path = "/home/natant/Thesis-plmbind/Data/Genomes/hg38.2bit" 
mask_path = "/home/data/shared/natant/Data/All_merged_ONLY_REG_OVERLAP.bed"
out_path = "/home/data/shared/natant/Data/testing_mask.h5t"
chrom = [1]
all_TFs = "/home/natant/Thesis-plmbind/Thesis/utils/TF_split/TF_data_split/ALL_TFs.txt"
alts = False

NOTchrom = []
# parser = argparse.ArgumentParser()
# parser.add_argument("bed_path")
# parser.add_argument("genome_path")
# parser.add_argument("mask_path")
# parser.add_argument("out_path")
# parser.add_argument('-chrom', nargs="+", type=int)
# parser.add_argument('-NOTchrom', nargs="+", type=int)
# parser.add_argument('-alts')
# parser.add_argument("-all_TFs", default="/home/natant/Thesis-plmbind/Thesis/utils/TF_split/TF_data_split/ALL_TFs.txt")
# args = parser.parse_args()
data = pd.read_csv(bed_path, sep = "\t", header = None)
mask_data = pd.read_csv(mask_path, sep = "\t", header = None)

TF_list = []
with open(all_TFs) as f:
    for TF in f:
        TF_list.append(TF.strip())
              
prot_db = dict(zip(TF_list, range(len(TF_list))))

tb = py2bit.open(genome_path)

# define mapping for nucleotides:
mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
# define mapping for proteins and celltypes, these will be filled in as we go:


def get_chrom_number(chr):
    try:
        num = int(chr[3:])
    except:
        num = 100
    return num

Unique_chrom_names = list(tb.chroms())
chrom_order = sorted(list(Unique_chrom_names), key=lambda x: get_chrom_number(x))
print("Used chromosomes: ")
if chrom != None:
    chroms = [chrom_order[i-1] for i in chrom]
    if alts == "True":
        chroms = [i for i in chrom_order if i.split("_")[0] in chroms]
    print(chroms)
elif NOTchrom != None:
    NOTchroms = [chrom_order[i-1] for i in NOTchrom]
    special_chroms = [i for i in chrom_order if i.split("_")[0] in NOTchroms]
    NOTchroms = NOTchroms + special_chroms
    chroms_w_alts = [i for i in chrom_order if i not in NOTchroms]
    chroms = [i for i in chroms_w_alts if len(i.split("_"))==1]
    if alts == "True":
        chroms = chroms_w_alts[:]
        
    print(chroms)
else:
    raise SystemExit(1)


# initialize final data objects:
sequence_per_chrom = []
rows_per_chrom = []
cols_per_chrom = []
rows_per_chrom_mask = []
cols_per_chrom_mask = []
for chrom_name in chroms:
    chrom_len = tb.chroms()[chrom_name]
    print(chrom_name)
    print('gettin chrom ..')
    # 1: get chrom
    chrom = np.array([mapping[l] for l in tb.sequence(chrom_name)], dtype=np.int8)

    #2: get peaks for chrom
    data_subset = data[data[0] == chrom_name]
    
    n_rows_chrom = (chrom_len//128) + 1

    rows = []
    cols = []
    for ix, i in enumerate(range(len(data_subset))):
        # Voor elke lijn in de bedfile voor dat specifiek chromosome
        start, stop, d = data_subset.iloc[i][[1, 2, 3]] # start stop en info
        prot, celltypes = d.split(':') # split info in TF en celltype
        start, stop, prot

        r = np.arange(start // 128, (stop // 128)+1) # go from 1 bp resolution to 128 bp resolution, so a region of 400 or whatever gets changed to 4 regions for 128 bp
        rows.append(r) # add these new coordinates to rows
        cols.append([prot_db[prot]] * len(r)) # for each of these positions add that the specific TF (ID) is positive.

    cols_per_chrom.append(np.concatenate(cols)) # add to the per chromosome versions
    rows_per_chrom.append(np.concatenate(rows))

    sequence_per_chrom.append(np.concatenate([
        chrom, # DNA of the chromosome
        np.array([4] * (((chrom_len//128) + 1) * 128 - chrom_len), dtype = np.int8), # add N's to the end so the total sequence length is a multiple of 128
    ]).reshape(-1, 128))

    # MASK
    mask_subset = mask_data[mask_data[0] == chrom_name]
    rows = []
    cols = []
    for ix, i in enumerate(range(len(mask_subset))):
        start, stop, d = data_subset.iloc[i][[1, 2, 3]]
        r = np.arange(start // 128, (stop // 128)+1)
        rows.append(r) # add these new coordinates to rows
    rows_per_chrom_mask.append(np.concatenate(rows))


lens = [s.shape[0] for s in sequence_per_chrom]
cumlens = np.cumsum([0] + lens)
y = np.zeros((sum(lens), len(prot_db)), dtype=np.bool_) # make large zero array
for r, c, add in zip(rows_per_chrom, cols_per_chrom, cumlens):
    y[r+add, c] = True # add those positions that are positive
from scipy.sparse import csr_matrix
mat = csr_matrix(y) # make it sparse
mat.indices = mat.indices.astype(np.int16)

# mask 
mask = np.zeros(sum(lens), dtype=np.bool_)
for r, add in zip(rows_per_chrom_mask, cumlens):
    mask[r+add] = True
mask_sparse = csr_matrix(np.expand_dims(mask, axis=1))



f = h5torch.File(out_path, "w")
f.register(mat, axis = "central", mode = "csr", dtype_save = "bool", dtype_load = "int64")

f.register(mask_sparse, axis = 0, name = "mask", mode = "csr", dtype_save = "bool", dtype_load = "int64")

f.register(np.concatenate(sequence_per_chrom), axis = 0, name = "DNA", mode = "N-D", dtype_save = "int8", dtype_load = "int64")
ix_to_prot = {v : k for k, v in prot_db.items()}
prot_mapping = np.array([ix_to_prot[i] for i in range(len(prot_db))]).astype(bytes)
f.register(prot_mapping, axis = 1, name = "prots", mode = "N-D", dtype_save = "bytes", dtype_load = "str")
chrom_lens = np.array([s.shape[0] for s in sequence_per_chrom])
f.register(chrom_lens, axis = "unstructured", name = "chrom_lens", mode = "N-D", dtype_save = "int64", dtype_load = "int64")
ix_to_nuc = {v : k for k, v in mapping.items()}
nucleotide_mapping = np.array([ix_to_nuc[i] for i in range(len(ix_to_nuc))]).astype(bytes)
f.register(nucleotide_mapping, axis = "unstructured", name = "nucleotide_mapping", mode = "N-D", dtype_save = "bytes", dtype_load = "str")

f.visititems(print)
f.close()
#########################################################


