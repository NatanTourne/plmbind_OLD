import numpy as np
import h5torch
import torch
from Bio import SeqIO
from tqdm import tqdm

seqs_path = "/home/natant/Thesis-plmbind/Testing_ground/TF_sequences.fasta"
sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}
h5_path = "/home/data/shared/natant/Data/test_no_alts.h5t"
f = h5torch.File(h5_path, "a")

n = 3
emb_name = "3mer_pad_trun"
TF_names = f["1/prots"][:].astype("str")

embedding_list = []
uniques = []
max_len = round(1024/n) # to take into account the same length as with the ESM embeddings
for TF in tqdm(list(TF_names), desc="splitting"):
    protein_kmer = [sequences[TF][i:i+n] for i in range(0, len(sequences[TF]), n) if len(sequences[TF][i:i+n])==n]
    [uniques.append(kmers) for kmers in protein_kmer if kmers not in uniques]
kmer_to_int_dict = {kmer:i+1 for i,kmer in enumerate(uniques)}
int_to_kmer_dict = {i+1:kmer for i,kmer in enumerate(uniques)}
for TF in tqdm(list(TF_names), desc="splitting"):
    protein_kmer = [kmer_to_int_dict[sequences[TF][i:i+n]] for i in range(0, len(sequences[TF]), n) if len(sequences[TF][i:i+n])==n]
    protein_kmer = torch.LongTensor(protein_kmer[:max_len])
    if protein_kmer.shape[0] < max_len:
        protein_kmer = torch.cat([protein_kmer, torch.tensor([0]*(max_len-protein_kmer.shape[0]))])
    embedding_list.append(np.array(protein_kmer))


f.register(np.array(embedding_list), axis = "unstructured", mode="separate", name=emb_name, dtype_save = "int32", dtype_load = "int32")
f.close()
print("done")

h5_path = "/home/data/shared/natant/Data/val_no_alts.h5t"
f = h5torch.File(h5_path, "a")
f.register(np.array(embedding_list), axis = "unstructured", mode="separate", name=emb_name, dtype_save = "int32", dtype_load = "int32")
f.close()
print("done")

h5_path = "/home/data/shared/natant/Data/train_no_alts.h5t"
f = h5torch.File(h5_path, "a")
f.register(np.array(embedding_list), axis = "unstructured", mode="separate", name=emb_name, dtype_save = "int32", dtype_load = "int32")
f.close()
print("done")
