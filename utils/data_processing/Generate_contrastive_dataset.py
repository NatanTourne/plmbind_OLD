import argparse
import pandas as pd
import numpy as np
import h5torch
import py2bit
import random

def get_chrom_number(chr):
    try:
        num = int(chr[3:])
    except:
        num = 100
    return num

# INPUTS
parser = argparse.ArgumentParser()
parser.add_argument("bed_path")
parser.add_argument("genome_path")
parser.add_argument("out_path")
parser.add_argument('-chrom', nargs="+", type=int)
parser.add_argument('-NOTchrom', nargs="+", type=int)
parser.add_argument('-alts')
parser.add_argument("-all_TFs", default="/home/natant/Thesis-plmbind/Thesis/utils/TF_split/TF_data_split/ALL_TFs.txt")
parser.add_argument("-length", default = 2048)
args = parser.parse_args()
data = pd.read_csv(args.bed_path, sep = "\t", header = None)
data.head()

desired_length = args.length
# CREATE TF library, Do it here so the indexes will be the same no matter the order they appear in the bed files
TF_list = []
with open(args.all_TFs) as f:
    for TF in f:
        TF_list.append(TF.strip())
              
prot_db = dict(zip(TF_list, range(len(TF_list))))

# load the genome
tb = py2bit.open(args.genome_path)

# define mapping for nucleotides:
mapping = {"A": 0, "T": 1, "C": 2, "G": 3, "N": 4}

# Define which chromosomes should be included in the dataset
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

# Actually fetch the data
sequence_per_chrom = []
rows_per_chrom = []
cols_per_chrom = []
DNA_frames = []
rows = []
cols = []
for chrom_name in chroms: # Go through each of the chromosomes that were selected
    chrom_rows = []
    print(chrom_name)
    print('gettin chrom ..')
    # get peaks for chrom
    data_subset = data[data[0] == chrom_name]
  
    
    for ix, i in enumerate(range(len(data_subset))):
        start, stop, TFs = data_subset.iloc[i][[1, 2, 3]]
        delta = stop-start
        if delta <= desired_length:
            if delta < desired_length: # If the peak is to short, add a prefix and suffix so its the right size
                prefix = random.randint(0,desired_length-delta)
                suffix = desired_length-delta-prefix
                start = start - prefix
                stop = stop + suffix
            
            # add the peak and TFs to the list
            chrom_rows.append((chrom_name, start, stop))
            cols.append([prot_db[i] for i in TFs.split(",")])
    # fetch the DNA sequences corresponding to the indexes        
    chrom = np.array([mapping[l] for l in tb.sequence(chrom_name)], dtype=np.int8)
    DNA_frames.extend([chrom[i[1]:i[2]] for i in chrom_rows])
    rows.extend(chrom_rows)
    
# Create the data matrix from the indices
y = np.zeros((len(rows), len(prot_db)), dtype=np.bool_)
y.shape
for i,TFs in enumerate(cols):
    y[i,TFs] = True
    
from scipy.sparse import csr_matrix
mat = csr_matrix(y) # Die matrix dan sparse maken
mat.indices = mat.indices.astype(np.int16)  

# create the datafile
f = h5torch.File(args.out_path, "w")
f.register(mat, axis="central", mode = "csr", dtype_save = "bool", dtype_load="int64")
f.register(np.array(DNA_frames), axis = 0, name = "DNA", mode = "N-D", dtype_save = "int8", dtype_load="int64")
ix_to_prot = {v : k for k, v in prot_db.items()}
prot_mapping = np.array([ix_to_prot[i] for i in range(len(prot_db))]).astype(bytes)
f.register(prot_mapping, axis = 1, name = "prots", mode = "N-D", dtype_save = "bytes", dtype_load = "str")
ix_to_nuc = {v : k for k, v in mapping.items()}
nucleotide_mapping = np.array([ix_to_nuc[i] for i in range(len(ix_to_nuc))]).astype(bytes)
f.register(nucleotide_mapping, axis = "unstructured", name = "nucleotide_mapping", mode = "N-D", dtype_save = "bytes", dtype_load = "str")

f.visititems(print)
f.close()
