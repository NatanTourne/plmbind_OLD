import numpy as np
import h5torch
import torch
from Bio import SeqIO
from io import StringIO
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import EsmModel, EsmConfig, EsmTokenizer
import requests
from tqdm import tqdm, trange

used_model = "facebook/esm2_t6_8M_UR50D"
h5_path = "/home/natant/Thesis/Data/Not_used/ReMap_testing_2/train_no_alts.h5t"
seqs_path = "/home/natant/Thesis/Testing_ground/sequences.fasta"
embeddings_name = "t6_320_pad_trun"
f = h5torch.File(h5_path, "a")

sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}

tokenizer = EsmTokenizer.from_pretrained(used_model)
model = EsmModel.from_pretrained(used_model)
TF_names = f["unstructured/protein_map"][:].astype("str")
embedding_list = []
for TF in tqdm(TF_names, desc="Fetching TF embeddings"):
    protein_seq = sequences[TF]
    inputs = tokenizer(protein_seq, return_tensors="pt", padding = True, truncation = True)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[0].detach().numpy()
    embedding_list.append(last_hidden_states)    

embeddings = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in  embedding_list]).permute(1,0,2)

f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")

used_model = "facebook/esm2_t6_8M_UR50D"
h5_path = "/home/natant/Thesis/Data/Not_used/ReMap_testing_2/test_no_alts.h5t"
seqs_path = "/home/natant/Thesis/Testing_ground/sequences.fasta"
embeddings_name = "t6_320_pad_trun"
f = h5torch.File(h5_path, "a")

sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}

tokenizer = EsmTokenizer.from_pretrained(used_model)
model = EsmModel.from_pretrained(used_model)
TF_names = f["unstructured/protein_map"][:].astype("str")
embedding_list = []
for TF in tqdm(TF_names, desc="Fetching TF embeddings"):
    protein_seq = sequences[TF]
    inputs = tokenizer(protein_seq, return_tensors="pt", padding = True, truncation = True)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[0].detach().numpy()
    embedding_list.append(last_hidden_states)    

embeddings = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in  embedding_list]).permute(1,0,2)

f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")

used_model = "facebook/esm2_t6_8M_UR50D"
h5_path = "/home/natant/Thesis/Data/Not_used/ReMap_testing_2/val_no_alts.h5t"
seqs_path = "/home/natant/Thesis/Testing_ground/sequences.fasta"
embeddings_name = "t6_320_pad_trun"
f = h5torch.File(h5_path, "a")

sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}

tokenizer = EsmTokenizer.from_pretrained(used_model)
model = EsmModel.from_pretrained(used_model)
TF_names = f["unstructured/protein_map"][:].astype("str")
embedding_list = []
for TF in tqdm(TF_names, desc="Fetching TF embeddings"):
    protein_seq = sequences[TF]
    inputs = tokenizer(protein_seq, return_tensors="pt", padding = True, truncation = True)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[0].detach().numpy()
    embedding_list.append(last_hidden_states)    

embeddings = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in  embedding_list]).permute(1,0,2)

f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")