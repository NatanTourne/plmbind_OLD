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

used_model = "facebook/esm2_t6_8M_UR50D" #"facebook/esm2_t12_35M_UR50D" #"facebook/esm2_t30_150M_UR50D" #"facebook/esm2_t6_8M_UR50D"
h5_path = "/home/data/shared/natant/Data/contrastive_train_no_alts.h5t"
seqs_path = "/home/natant/Thesis-plmbind/Testing_ground/sequences.fasta"
embeddings_name = "t6_320_pad_trun"#"t30_640_pad_trun" # "t12_480_pad_trun" # "t6_320_pad_trun" # "t33_1280_pad_trun"
f = h5torch.File(h5_path, "a")

sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}

tokenizer = EsmTokenizer.from_pretrained(used_model)
model = EsmModel.from_pretrained(used_model).to("cuda:1")
TF_names = f["1/prots"][:].astype("str")
embedding_list = []
for TF in tqdm(TF_names, desc="Fetching TF embeddings"):
    protein_seq = sequences[TF]
    inputs = tokenizer(protein_seq, return_tensors="pt", padding = 'max_length', truncation = True, max_length=1024).to("cuda:1")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[0].detach().cpu().numpy()
    embedding_list.append(last_hidden_states)    

embeddings = torch.nn.utils.rnn.pad_sequence([torch.tensor(i) for i in  embedding_list]).permute(1,0,2)

f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")

h5_path = "/home/data/shared/natant/Data/contrastive_test_no_alts.h5t"
f = h5torch.File(h5_path, "a")
f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")

h5_path = "/home/data/shared/natant/Data/contrastive_val_no_alts.h5t"
f = h5torch.File(h5_path, "a")
f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")
