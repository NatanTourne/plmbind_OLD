import numpy as np
import h5torch
import torch
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
embedding_name = "3mers_pad_trunc"

seqs_path = "/home/natant/Thesis-plmbind/Testing_ground/TF_sequences.fasta"
sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}
h5_path = "/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/train_no_alts.h5t"
f = h5torch.File(h5_path, "a")
TF_names = f["unstructured/protein_map"][:].astype("str")

embedding_list = []
uniques = []
n = 3
max_len = round(1024/n) # to take into account the same length as with the ESM embeddings
for TF in tqdm(list(TF_names), desc="splitting"):
    protein_kmer = [sequences[TF][i:i+n] for i in range(0, len(sequences[TF]), n) if len(sequences[TF][i:i+n])==n]
    [uniques.append(kmers) for kmers in protein_kmer if kmers not in uniques]
kmer_to_int_dict = {kmer:i+1 for i,kmer in enumerate(uniques)}
int_to_kmer_dict = {i+1:kmer for i,kmer in enumerate(uniques)}
for TF in tqdm(list(TF_names), desc="splitting"):
    protein_kmer = [kmer_to_int_dict[sequences[TF][i:i+n]] for i in range(0, len(sequences[TF]), n) if len(sequences[TF][i:i+n])==n]
    protein_kmer = torch.LongTensor(protein_kmer[:max_len])
    #protein_kmer = torch.nn.functional.pad(protein_kmer, (0,max_len-protein_kmer.shape[0]))
    embedding_list.append(protein_kmer)#torch.nn.functional.pad(protein_kmer, (0,max_len-protein_kmer.shape[0])))
embedding_list = [torch.nn.functional.one_hot(i, num_classes=len(kmer_to_int_dict)+1) for i in embedding_list]
embedding_list_padded = torch.nn.utils.rnn.pad_sequence([i for i in  embedding_list]).permute(1,0,2)

f.register(np.array(embedding_list_padded), axis = "unstructured", mode="separate", name=embedding_name, dtype_save = "float32", dtype_load = "float32")
f.close()

seqs_path = "/home/natant/Thesis-plmbind/Testing_ground/TF_sequences.fasta"
sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}
h5_path = "/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/test_no_alts.h5t"
f = h5torch.File(h5_path, "a")
TF_names = f["unstructured/protein_map"][:].astype("str")

embedding_list = []
uniques = []
n = 3
max_len = round(1024/n) # to take into account the same length as with the ESM embeddings
for TF in tqdm(list(TF_names), desc="splitting"):
    protein_kmer = [sequences[TF][i:i+n] for i in range(0, len(sequences[TF]), n) if len(sequences[TF][i:i+n])==n]
    [uniques.append(kmers) for kmers in protein_kmer if kmers not in uniques]
kmer_to_int_dict = {kmer:i+1 for i,kmer in enumerate(uniques)}
int_to_kmer_dict = {i+1:kmer for i,kmer in enumerate(uniques)}
for TF in tqdm(list(TF_names), desc="splitting"):
    protein_kmer = [kmer_to_int_dict[sequences[TF][i:i+n]] for i in range(0, len(sequences[TF]), n) if len(sequences[TF][i:i+n])==n]
    protein_kmer = torch.LongTensor(protein_kmer[:max_len])
    #protein_kmer = torch.nn.functional.pad(protein_kmer, (0,max_len-protein_kmer.shape[0]))
    embedding_list.append(protein_kmer)#torch.nn.functional.pad(protein_kmer, (0,max_len-protein_kmer.shape[0])))
embedding_list = [torch.nn.functional.one_hot(i, num_classes=len(kmer_to_int_dict)+1) for i in embedding_list]
embedding_list_padded = torch.nn.utils.rnn.pad_sequence([i for i in  embedding_list]).permute(1,0,2)

f.register(np.array(embedding_list_padded), axis = "unstructured", mode="separate", name=embedding_name, dtype_save = "float32", dtype_load = "float32")
f.close()

seqs_path = "/home/natant/Thesis-plmbind/Testing_ground/TF_sequences.fasta"
sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}
h5_path = "/home/natant/Thesis-plmbind/Data/Not_used/ReMap_testing_2/val_no_alts.h5t"
f = h5torch.File(h5_path, "a")
TF_names = f["unstructured/protein_map"][:].astype("str")

embedding_list = []
uniques = []
n = 3
max_len = round(1024/n) # to take into account the same length as with the ESM embeddings
for TF in tqdm(list(TF_names), desc="splitting"):
    protein_kmer = [sequences[TF][i:i+n] for i in range(0, len(sequences[TF]), n) if len(sequences[TF][i:i+n])==n]
    [uniques.append(kmers) for kmers in protein_kmer if kmers not in uniques]
kmer_to_int_dict = {kmer:i+1 for i,kmer in enumerate(uniques)}
int_to_kmer_dict = {i+1:kmer for i,kmer in enumerate(uniques)}
for TF in tqdm(list(TF_names), desc="splitting"):
    protein_kmer = [kmer_to_int_dict[sequences[TF][i:i+n]] for i in range(0, len(sequences[TF]), n) if len(sequences[TF][i:i+n])==n]
    protein_kmer = torch.LongTensor(protein_kmer[:max_len])
    #protein_kmer = torch.nn.functional.pad(protein_kmer, (0,max_len-protein_kmer.shape[0]))
    embedding_list.append(protein_kmer)#torch.nn.functional.pad(protein_kmer, (0,max_len-protein_kmer.shape[0])))
embedding_list = [torch.nn.functional.one_hot(i, num_classes=len(kmer_to_int_dict)+1) for i in embedding_list]
embedding_list_padded = torch.nn.utils.rnn.pad_sequence([i for i in  embedding_list]).permute(1,0,2)

f.register(np.array(embedding_list_padded), axis = "unstructured", mode="separate", name=embedding_name, dtype_save = "float32", dtype_load = "float32")
f.close()