import numpy as np
import h5torch
from Bio import SeqIO
from io import StringIO
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import EsmModel, EsmConfig, EsmTokenizer
import requests
from tqdm import tqdm, trange

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
h5_path = "/home/natant/Thesis/Data/ReMap_testing/train.h5t"
out_path = "/home/natant/Thesis/plmbind/data_processing/TF_sequences.fasta"
f = h5torch.File(h5_path, "a")
TF_names = f["unstructured/protein_map"][:].astype("str")
f.close()
protein_seqs = []
for TF in tqdm(TF_names, desc="Fetching TF sequences"):
    protein_seqs.append(str(get_protein_sequence(TF)))
    
with open(out_path, 'w') as f:
    for i, seq in enumerate(protein_seqs):
        f.write(">" + TF_names[i] + "\n" + seq + "\n")
    