#!/bin/bash

echo "Generating Test dataset"
python Generate_dataset.py /home/natant/Thesis/Data/ReMap2022/remap2022_nr_macs2_hg38_v1_0.bed /home/natant/Thesis/Data/Genomes/hg38.2bit /home/natant/Thesis/Data/ReMap2022/test.h5t -chrom 1 2 3 4

echo "Generating Validation dataset"
python Generate_dataset.py /home/natant/Thesis/Data/ReMap2022/remap2022_nr_macs2_hg38_v1_0.bed /home/natant/Thesis/Data/Genomes/hg38.2bit /home/natant/Thesis/Data/ReMap2022/val.h5t -chrom 5 6 7 8

echo "Generating Training dataset"
python Generate_dataset.py /home/natant/Thesis/Data/ReMap2022/remap2022_nr_macs2_hg38_v1_0.bed /home/natant/Thesis/Data/Genomes/hg38.2bit /home/natant/Thesis/Data/ReMap2022/train.h5t -NOTchrom 1 2 3 4 5 6 7 8

echo "Adding protein embeddings to Test dataset"
python add_protein_embeddings.py /home/natant/Thesis/Data/ReMap2022/test.h5t facebook/esm2_t6_8M_UR50D prot_embeddings_t6

echo "Adding protein embeddings to Validation dataset"
python add_protein_embeddings.py /home/natant/Thesis/Data/ReMap2022/val.h5t facebook/esm2_t6_8M_UR50D prot_embeddings_t6

echo "Adding protein embeddings to training dataset"
python add_protein_embeddings.py /home/natant/Thesis/Data/ReMap2022/train.h5t facebook/esm2_t6_8M_UR50D prot_embeddings_t6