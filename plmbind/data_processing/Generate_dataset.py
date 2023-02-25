import argparse
import pandas as pd
import numpy as np
import operator
parser = argparse.ArgumentParser()
parser.add_argument("bed_path")
parser.add_argument("genome_path")
parser.add_argument("out_path")
parser.add_argument('-chrom', nargs="+", type=int)
parser.add_argument('-NOTchrom', nargs="+", type=int)
parser.add_argument('-alts')
args = parser.parse_args()
data = pd.read_csv(args.bed_path, sep = "\t", header = None)
data.head()

import py2bit
tb = py2bit.open(args.genome_path)

# define mapping for nucleotides:
mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}
# define mapping for proteins and celltypes, these will be filled in as we go:
prot_db = {}
celltype_db = {}
enum_prots = 0
enum_celltypes = 0
enum_prot_celltype_comb = 0

# initialize final data objects:
sequence_per_chrom = []
chrom_lens = []
peaks_per_chrom_prot = []
peaks_per_chrom_celltype = []
peaks_per_chrom_start = []
peaks_per_chrom_end = []


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

for chrom_name in chroms:
    chrom_len = tb.chroms()[chrom_name]
    print(chrom_name)
    print('gettin chrom ..')
    # 1: get chrom
    chrom = np.array([mapping[l] for l in tb.sequence(chrom_name)], dtype=np.int8)
    sequence_per_chrom.append(chrom)

    #2: get peaks for chrom
    data_subset = data[data[0] == chrom_name]

    #3: create a list of pairs [(P_x, C_y), (...)] for every peak
    peaks_prot = []
    peaks_celltype = []
    peaks_start = []
    peaks_end = []
    print('gettin peaks ..')
    for i in range(len(data_subset)):
        start, stop, d = data_subset.iloc[i][[1, 2, 3]]
        prot, celltypes = d.split(':')
        celltypes = celltypes.split(',')

        # register new prots and celltypes if needed:
        if prot not in prot_db:
            prot_db[prot] = enum_prots
            enum_prots += 1

        for celltype in celltypes:
            if celltype not in celltype_db:
                celltype_db[celltype] = enum_celltypes
                enum_celltypes += 1
        
        for ct in celltypes:
            peaks_prot.append(prot_db[prot])
            peaks_celltype.append(celltype_db[ct])
            peaks_start.append(start)
            peaks_end.append(stop)

        if i % 25000 == 0:
            print("peak", i, "..")

        
    peaks_per_chrom_prot.append(np.array(peaks_prot, dtype = np.int16))
    peaks_per_chrom_celltype.append(np.array(peaks_celltype, dtype = np.int16))
    peaks_per_chrom_start.append(np.array(peaks_start, dtype = np.int64))
    peaks_per_chrom_end.append(np.array(peaks_end, dtype = np.int64))

    # keep track of chrom lens
    chrom_lens.append(chrom_len)
    
for i, sum_to in zip(range(len(peaks_per_chrom_start)), np.cumsum([0] + list((np.array(chrom_lens) + 10_000)))):
    peaks_per_chrom_start[i] = peaks_per_chrom_start[i] + sum_to
    peaks_per_chrom_end[i] = peaks_per_chrom_end[i] + sum_to
    
peaks_per_chrom_end = np.concatenate(peaks_per_chrom_end)
peaks_per_chrom_start = np.concatenate(peaks_per_chrom_start)
peaks_per_chrom_prot = np.concatenate(peaks_per_chrom_prot)
peaks_per_chrom_celltype = np.concatenate(peaks_per_chrom_celltype)

all_sequences = np.empty(0)
for seq in sequence_per_chrom:
    all_sequences = np.concatenate([all_sequences, seq])
    all_sequences = np.concatenate([all_sequences, np.ones(10000, dtype=np.int8) * 5])
all_sequences = all_sequences[:-10000]

map_rev = {v : k for k, v in mapping.items()}
nucleotide_mapping = np.array([map_rev[i] for i in range(len(map_rev))])

map_rev = {v : k for k, v in prot_db.items()}
protein_mapping = np.array([map_rev[i] for i in range(len(map_rev))])

map_rev = {v : k for k, v in celltype_db.items()}
celltype_mapping = np.array([map_rev[i] for i in range(len(map_rev))])

starts = peaks_per_chrom_start // 100_000
ends = peaks_per_chrom_end // 100_000
slice_indices_per_100000 = []
for i in range(all_sequences.shape[0] // 100_000):
    ends_in_i = np.where(i == ends)[0]
    starts_in_i = np.where(i == starts)[0]
    if (ends_in_i.shape[0] != 0) and (starts_in_i.shape[0] != 0):
        start_ix_i = ends_in_i[0]
        end_ix_i = starts_in_i[-1] + 1
    elif (ends_in_i.shape[0] == 0) and (starts_in_i.shape[0] == 0):
        if len(slice_indices_per_100000) > 0:
            start_ix_i = slice_indices_per_100000[-1][-1]
            end_ix_i = slice_indices_per_100000[-1][-1]
        else:
            start_ix_i = 0
            end_ix_i = 0
    elif (ends_in_i.shape[0] == 0) and (starts_in_i.shape[0] != 0):
        end_ix_i = starts_in_i[-1] + 1
        start_ix_i = end_ix_i - 1
    elif (starts_in_i.shape[0] == 0) and (ends_in_i.shape[0] != 0):
        start_ix_i = ends_in_i[0]
        end_ix_i = start_ix_i + 1

    slice_indices_per_100000.append([start_ix_i, end_ix_i])

slice_indices_per_100000 = np.array(slice_indices_per_100000)

import h5torch

f = h5torch.File(args.out_path, "w")
f.register(all_sequences, axis = "central", mode = "N-D", name = "seq", dtype_save = "int8", dtype_load = "int64")

f.register(peaks_per_chrom_start, name = "peaks_start", axis = "unstructured", dtype_save = "int64", dtype_load = "int64")
f.register(peaks_per_chrom_end, name = "peaks_end", axis = "unstructured", dtype_save = "int64", dtype_load = "int64")
f.register(peaks_per_chrom_prot, name = "peaks_prot", axis = "unstructured", dtype_save = "int16", dtype_load = "int64")
f.register(peaks_per_chrom_celltype, name = "peaks_celltypes", axis = "unstructured", dtype_save = "int16", dtype_load = "int64")

f.register(slice_indices_per_100000, name = "slice_indices_per_100000", axis = "unstructured", dtype_save = "int64", dtype_load = "int64")


f.register(nucleotide_mapping, name = "nucleotide_map", axis = "unstructured", dtype_save = "bytes", dtype_load="str")
f.register(protein_mapping, name = "protein_map", axis = "unstructured", dtype_save = "bytes", dtype_load="str")
f.register(celltype_mapping, name = "celltype_map", axis = "unstructured", dtype_save = "bytes", dtype_load="str")


f.register(np.array(chrom_lens), name = "chrom_lens", axis = "unstructured", dtype_save = "int64", dtype_load = "int64")

f.visititems(print)
f.close()