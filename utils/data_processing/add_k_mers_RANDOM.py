import numpy as np
import h5torch
import torch
from Bio import SeqIO
from tqdm import tqdm

seqs_path = "/home/natant/Thesis-plmbind/Thesis/utils/data_processing/TF_sequences.fasta"
sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}
h5_path = "/home/data/shared/natant/Data/train_no_alts.h5t"
f = h5torch.File(h5_path, "a")

n = 3 #3
emb_name = "3mer_RANDOM" #"3mer_pad_trun"
TF_names = f["1/prots"][:].astype("str")


max_len = round(1024/n) # to take into account the same length as with the ESM embeddings

embeddings = torch.randint(1, 7917, (len(TF_names), max_len))
f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=emb_name, dtype_save = "int32", dtype_load = "int32")
f.close()
print("done")

h5_path = "/home/data/shared/natant/Data/val_no_alts.h5t"
f = h5torch.File(h5_path, "a")
f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=emb_name, dtype_save = "int32", dtype_load = "int32")
f.close()
print("done")

h5_path = "/home/data/shared/natant/Data/test_no_alts.h5t"
f = h5torch.File(h5_path, "a")
f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=emb_name, dtype_save = "int32", dtype_load = "int32")
f.close()
print("done")
