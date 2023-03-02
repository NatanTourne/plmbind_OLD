import argparse
import numpy as np
import h5torch
from Bio import SeqIO
from io import StringIO
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import EsmModel, EsmConfig, EsmTokenizer
import requests

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("embeddings")
parser.add_argument("embeddings_name")
args = parser.parse_args()

def get_protein_sequence(TF):
    r = requests.get(f'http://remap.univ-amu.fr/api/v1/info/byTarget/target={TF}&taxid=9606')
    if r.status_code != 200:
        return np.array(0)
    else:
        if r.json()["external_IDs"]["others"]=='':
            return np.array(0)
        else:
            uniprot_ID = r.json()["external_IDs"]["others"]["Uniprot"]
            r1 = requests.get(f"https://www.uniprot.org/uniprot/{uniprot_ID}.fasta")
            fasta_sequences = SeqIO.parse(StringIO(r1.text),'fasta')
            protein_sequence = list(fasta_sequences)[0].seq
            return protein_sequence
        
f = h5torch.File(args.path, "a")

embeddings = args.embeddings
# embeddings = "facebook/esm2_t33_650M_UR50D"
# Protein features
tokenizer = EsmTokenizer.from_pretrained(embeddings)
model = EsmModel.from_pretrained(embeddings)
TF_names = f["unstructured/protein_map"][:].astype("str")
embedding_list = []
for TF in TF_names:
    print(TF)
    protein_seq = str(get_protein_sequence(TF))
    inputs = tokenizer(protein_seq, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state[0].detach().mean(axis=0).numpy()
    embedding_list.append(last_hidden_states)
    
f.register(np.array(embedding_list), axis = "unstructured", mode="separate", name=args.embeddings_name, dtype_save = "float32", dtype_load = "float32")
f.close()
print("done")