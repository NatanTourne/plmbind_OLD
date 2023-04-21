import argparse
import pandas as pd
import numpy as np
import h5torch
import py2bit

parser = argparse.ArgumentParser()
parser.add_argument("bed_path")
parser.add_argument("genome_path")
parser.add_argument("out_path")
parser.add_argument('-chrom', nargs="+", type=int)
parser.add_argument('-NOTchrom', nargs="+", type=int)
parser.add_argument('-alts')
parser.add_argument("-all_TFs", default="/home/natant/Thesis-plmbind/Thesis/utils/TF_split/TF_data_split/ALL_TFs.txt")
args = parser.parse_args()
data = pd.read_csv(args.bed_path, sep = "\t", header = None)
data.head()

TF_list = []
with open(args.all_TFs) as f:
    for TF in f:
        TF_list.append(TF.strip())
              
prot_db = dict(zip(TF_list, range(len(TF_list))))

tb = py2bit.open(args.genome_path)

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
if args.chrom != None:
    chroms = [chrom_order[i-1] for i in args.chrom]
    if args.alts == "True":
        chroms = [i for i in chrom_order if i.split("_")[0] in chroms]
    print(chroms)
elif args.NOTchrom != None:
    NOTchroms = [chrom_order[i-1] for i in args.NOTchrom]
    special_chroms = [i for i in chrom_order if i.split("_")[0] in NOTchroms]
    NOTchroms = NOTchroms + special_chroms
    chroms_w_alts = [i for i in chrom_order if i not in NOTchroms]
    chroms = [i for i in chroms_w_alts if len(i.split("_"))==1]
    if args.alts == "True":
        chroms = chroms_w_alts[:]
        
    print(chroms)
else:
    raise SystemExit(1)


# initialize final data objects:
sequence_per_chrom = []
rows_per_chrom = []
cols_per_chrom = []
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
        start, stop, d = data_subset.iloc[i][[1, 2, 3]]
        prot, celltypes = d.split(':')
        start, stop, prot

        r = np.arange(start // 128, (stop // 128)+1)
        rows.append(r)
        cols.append([prot_db[prot]] * len(r))

    cols_per_chrom.append(np.concatenate(cols))
    rows_per_chrom.append(np.concatenate(rows))

    sequence_per_chrom.append(np.concatenate([
        chrom,
        np.array([4] * (((chrom_len//128) + 1) * 128 - chrom_len), dtype = np.int8),
    ]).reshape(-1, 128))


lens = [s.shape[0] for s in sequence_per_chrom]
cumlens = np.cumsum([0] + lens)
y = np.zeros((sum(lens), len(prot_db)), dtype=np.bool_)
for r, c, add in zip(rows_per_chrom, cols_per_chrom, cumlens):
    y[r+add, c] = True
from scipy.sparse import csr_matrix
mat = csr_matrix(y)
mat.indices = mat.indices.astype(np.int16)

f = h5torch.File(args.out_path, "w")
f.register(mat, axis = "central", mode = "csr", dtype_save = "bool", dtype_load = "int64")
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


