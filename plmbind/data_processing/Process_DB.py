from matplotlib.dviread import Dvi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import h5torch
import requests
from Bio import SeqIO
from io import StringIO
from tqdm import tqdm

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
        
df = pd.read_csv(r"/home/natant/Thesis/Data/TF_database/DatabaseExtract_v_1.01.csv")
h5_path = "/home/natant/Thesis/Data/Not_used/ReMap_testing/train.h5t"
out_path = "/home/natant/Thesis/Data/TF_database/info.csv"
f = h5torch.File(h5_path, "a")
TF_list = f["unstructured/protein_map"][:].astype("str")
f.close()

DB_TF_only = df[df["Is TF?"]=="Yes"]["HGNC symbol"].to_list()
DB_all = df["HGNC symbol"].to_list()
print("TFs in ChIP dataset: " + str(len(TF_list)))
print("Amount found in TF database: "+str(len([TF for TF in TF_list if TF in DB_all])))
print("Amount identified as TF by database: "+str(len([TF for TF in TF_list if TF in DB_TF_only])))

filtered_TF_list = [TF for TF in TF_list if TF in DB_TF_only]
df_filtered = df[[TF2 in filtered_TF_list for TF2 in df["HGNC symbol"].to_list()]]

protein_seqs = []
for TF in tqdm(df_filtered["HGNC symbol"].to_list(), desc="Fetching TF sequences"):
    protein_seqs.append(str(get_protein_sequence(TF)))
df_filtered["Sequence"] = protein_seqs

df_filtered.to_csv(out_path)