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

h5_path = "/home/data/shared/natant/Data/train_no_alts.h5t"
seqs_path = "/home/natant/Thesis-plmbind/Testing_ground/sequences.fasta"
embeddings_name = "t6_320_RANDOM"#"t30_640_pad_trun" # "t12_480_pad_trun" # "t6_320_pad_trun"
f = h5torch.File(h5_path, "a")

sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}

TF_names = f["1/prots"][:].astype("str")

embeddings = torch.FloatTensor(len(TF_names), 1024, 320).uniform_(-5, 5)


f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")

h5_path = "/home/data/shared/natant/Data/test_no_alts.h5t"
f = h5torch.File(h5_path, "a")
f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")

h5_path = "/home/data/shared/natant/Data/val_no_alts.h5t"
f = h5torch.File(h5_path, "a")
f.register(np.array(embeddings), axis = "unstructured", mode="separate", name=embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")
