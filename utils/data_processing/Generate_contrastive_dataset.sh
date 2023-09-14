#!/bin/bash

# echo "Generating Test dataset"
# python /home/natant/Thesis-plmbind/Thesis/utils/data_processing/Generate_contrastive_dataset.py /home/data/shared/natant/Data/remap_merged_peaks.bed /home/natant/Thesis-plmbind/Data/Genomes/hg38.2bit /home/data/shared/natant/Data/contrastive_test_no_alts.h5t -chrom 1 2 3 4

# echo "Generating Validation dataset"
# python /home/natant/Thesis-plmbind/Thesis/utils/data_processing/Generate_contrastive_dataset.py /home/data/shared/natant/Data/remap_merged_peaks.bed /home/natant/Thesis-plmbind/Data/Genomes/hg38.2bit /home/data/shared/natant/Data/contrastive_val_no_alts.h5t -chrom 5 6 7 8

echo "Generating Training dataset"
python /home/natant/Thesis-plmbind/Thesis/utils/data_processing/Generate_contrastive_dataset.py /home/data/shared/natant/Data/remap_merged_peaks.bed /home/natant/Thesis-plmbind/Data/Genomes/hg38.2bit /home/data/shared/natant/Data/contrastive_train_no_alts.h5t -NOTchrom 1 2 3 4 5 6 7 8
