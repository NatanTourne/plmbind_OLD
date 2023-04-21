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

# def get_protein_sequence(TF):
#     r = requests.get(f'http://remap.univ-amu.fr/api/v1/info/byTarget/target={TF}&taxid=9606')
#     if r.status_code != 200:
#         return np.array(0)
#     else:
#         if r.json()["external_IDs"]["others"]=='':
#             return np.array(0)
#         else:
#             uniprot_ID = r.json()["external_IDs"]["others"]["Uniprot"]
#             r1 = requests.get(f"https://www.uniprot.org/uniprot/{uniprot_ID}.fasta")
#             fasta_sequences = SeqIO.parse(StringIO(r1.text),'fasta')
#             protein_sequence = list(fasta_sequences)[0].seq
#             return protein_sequence


df = pd.read_csv(r"/home/natant/Thesis-plmbind/Data/TF_database/DatabaseExtract_v_1.01.csv")
out_path = "/home/natant/Thesis-plmbind/Data/TF_database/info.csv"

TF_list = []
with open('/home/natant/Thesis-plmbind/Thesis/utils/TF_split/TF_data_split/ALL_TFs.txt') as f:
    for TF in f:
        TF_list.append(TF.strip())

DB_TF_only = df[df["Is TF?"]=="Yes"]["HGNC symbol"].to_list()
DB_all = df["HGNC symbol"].to_list()

filtered_TF_list = [TF for TF in TF_list if TF in DB_TF_only]

seqs_path = "/home/natant/Thesis-plmbind/Thesis/utils/data_processing/TF_sequences.fasta"
sequences = {fasta.id:str(fasta.seq) for fasta in SeqIO.parse(open(seqs_path),'fasta')}
filtered_TF_list_with_seq = [TF for TF in filtered_TF_list if sequences[TF] != "0"]

print("TFs in ChIP dataset: " + str(len(TF_list)))
print("Amount found in TF database: "+str(len([TF for TF in TF_list if TF in DB_all])))
print("Amount identified as TF by database: "+str(len([TF for TF in TF_list if TF in DB_TF_only])))
print("With sequence: "+str(len(filtered_TF_list_with_seq)))

df_filtered = df[[TF2 in filtered_TF_list_with_seq for TF2 in df["HGNC symbol"].to_list()]]

protein_seqs = []
for TF in tqdm(df_filtered["HGNC symbol"].to_list(), desc="Fetching TF sequences"):
    protein_seqs.append(str(sequences[TF]))
df_filtered["Sequence"] = protein_seqs

df_filtered.to_csv(out_path)