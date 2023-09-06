# plmbind

## Description
This project explores the addition of protein side information for the prediction of transcription factor binding sites (TFBSs). This repository contains a baseline model that does not utilize any protein side information and two-branch models that process the DNA info and TF info in dedicated branches. The TF side information can be provided in the form of k-mers or as embeddings creased by protein language models (plm). Esm2 was used in this case, but the model can easily be extended to accept other embeddings. The code for generating the dataset and data splits is also included. 

## Usage
The main models and dataloader is found in the folder "plmbind" along with a number of training scripts. The code necessary for generating the dataset and data splits can be found under "utils".

