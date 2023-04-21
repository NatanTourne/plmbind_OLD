import numpy as np
import h5torch
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from h5torch.dataset import apply_dtype
import random

class ReMapDataModule2_0(pl.LightningDataModule):
    def __init__(
        self,
        train_loc,
        val_loc,
        test_loc,
        TF_list,
        val_list,
        embeddings,
        TF_batch_size=0,
        window_size=8192,
        overlap=8,
        batch_size: int = 32,
        num_workers = 3
    ):
        super().__init__()
        self.train_loc = train_loc
        self.val_loc = val_loc
        self.test_loc = test_loc
        self.batch_size = batch_size
        self.window_size = window_size
        self.overlap = overlap
        self.shave_edges = overlap // 2
        self.TF_list = TF_list
        self.val_list = val_list
        self.TF_batch_size = TF_batch_size
        self.embeddings = embeddings
        self.num_workers = num_workers

        # TRAIN DATA
        ## GET INDICES
        f = h5torch.File(self.train_loc)
        cumchromlens = np.cumsum([0] + list(f["unstructured/chrom_lens"][:]))
        start_pos_each_chrom = cumchromlens[:-1]
        end_pos_each_chrom = cumchromlens[1:] # geen -10000 hier??

        windows_to_use_as_samples = []
        for starts, stops in zip(start_pos_each_chrom, end_pos_each_chrom): # chroms to use weg gedaan
            starts = np.arange(starts, stops, window_size/128 - overlap)[:-1]
            stops = starts+window_size/128

            window_indices = np.stack([starts, stops]).T
            windows_to_use_as_samples.append(window_indices)

        self.windows_to_use_as_samples_train = np.concatenate(windows_to_use_as_samples).astype(int)
        
        ## CREATE DATASET
        self.train_data = h5torch.SliceDataset(
            self.train_loc,
            window_indices = self.windows_to_use_as_samples_train,
            sample_processor=SamplePreprocessor(shave_edges = self.shave_edges, TFs = None))
        
        # VALIDATION DATA
        ## GET INDICES
        f = h5torch.File(self.val_loc)
        cumchromlens = np.cumsum([0] + list(f["unstructured/chrom_lens"][:]))
        start_pos_each_chrom = cumchromlens[:-1]
        end_pos_each_chrom = cumchromlens[1:] # geen -10000 hier??

        windows_to_use_as_samples = []
        for starts, stops in zip(start_pos_each_chrom, end_pos_each_chrom): # chroms to use weg gedaan
            starts = np.arange(starts, stops, window_size/128 - overlap)[:-1]
            stops = starts+window_size/128

            window_indices = np.stack([starts, stops]).T
            windows_to_use_as_samples.append(window_indices)

        self.windows_to_use_as_samples_val = np.concatenate(windows_to_use_as_samples).astype(int)
        
        ## CREATE DATASET
        self.val_data = h5torch.SliceDataset(
            self.val_loc,
            window_indices = self.windows_to_use_as_samples_val,
            sample_processor=SamplePreprocessor(shave_edges = self.shave_edges, TFs = None))
        
        # TEST DATA
        ## GET INDICES
        f = h5torch.File(self.test_loc)
        cumchromlens = np.cumsum([0] + list(f["unstructured/chrom_lens"][:]))
        start_pos_each_chrom = cumchromlens[:-1]
        end_pos_each_chrom = cumchromlens[1:] # geen -10000 hier??

        windows_to_use_as_samples = []
        for starts, stops in zip(start_pos_each_chrom, end_pos_each_chrom): # chroms to use weg gedaan
            starts = np.arange(starts, stops, window_size/128 - overlap)[:-1]
            stops = starts+window_size/128

            window_indices = np.stack([starts, stops]).T
            windows_to_use_as_samples.append(window_indices)

        self.windows_to_use_as_samples_test = np.concatenate(windows_to_use_as_samples).astype(int)
        
        ## CREATE DATASET
        self.test_data = h5torch.SliceDataset(
            self.test_loc,
            window_indices = self.windows_to_use_as_samples_test,
            sample_processor=SamplePreprocessor(shave_edges = self.shave_edges, TFs = None))


    def setup(self, stage=None):
        pass
            
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle = True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=Collater2_0(self.train_loc, self.TF_list, self.embeddings, TF_batch_size = self.TF_batch_size)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=Collater_val2_0(self.val_loc, self.TF_list, self.val_list, self.embeddings, TF_batch_size = self.TF_batch_size)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=Collater2_0(self.test_loc, self.TF_list, self.embeddings, TF_batch_size = self.TF_batch_size)
        )


    # how is the predict dataloader different from train/val/test // does this need to be here?
    def predict_setup(self, Predict_TF, Data_split):
        self.predict_TF = Predict_TF
        if Data_split.lower() == "train":
            self.pred_loc = self.train_loc
            self.windows_to_use_as_samples_pred = self.windows_to_use_as_samples_train
        elif Data_split.lower() == "val":
            self.pred_loc = self.val_loc
            self.windows_to_use_as_samples_pred = self.windows_to_use_as_samples_val
        elif Data_split.lower() == "test":
            self.pred_loc = self.test_loc
            self.windows_to_use_as_samples_pred = self.windows_to_use_as_samples_test
        else:
            raise Exception("Not a valid data split")
        self.predict_data = h5torch.SliceDataset(
            self.pred_loc,
            window_indices = self.windows_to_use_as_samples_pred,
            sample_processor=SamplePreprocessor(shave_edges = self.shave_edges, TFs = None))
    
    def predict_dataloader(self):
        return DataLoader(
            self.predict_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=Collater2_0(self.pred_loc, self.predict_TF, self.embeddings, TF_batch_size = self.TF_batch_size)
        )

   
class SamplePreprocessor(object):
    def __init__(self, shave_edges = 4, TFs = None): #TFs can be a numpy array selecting labels
        self.shave_edges = shave_edges
        self.TFs = TFs
    def __call__(self, h5, sample):
        y = sample["central"][self.shave_edges:-self.shave_edges].T
        DNA = sample["0/DNA"].reshape(-1)
        if self.TFs is not None:
            y = y[self.TFs]
        return DNA, y    

class Collater2_0():
    def __init__(self, path, TF_list, embeddings, TF_batch_size = 0):
        f = h5torch.File(path)
        self.TF_batch_size = TF_batch_size
        self.TF_list = TF_list
        self.protein_index_list = np.concatenate(
            [np.where(f["1/prots"][:].astype('str') == i) for i in self.TF_list]
            ).flatten()
        embedding_list_temp = []
        for i in self.protein_index_list:
            embedding_list_temp.append(
                f[embeddings][str(i)][:]
                )
        self.embedding_list = embedding_list_temp[:]

    def __call__(self, batch):
        # SAMPLE TFs (Too many cause memory issues)
        Used_index = []
        Used_embeddings = []
        if self.TF_batch_size == 0:
            Used_index = self.protein_index_list[:]
            Used_embeddings = self.embedding_list
        else:
            samples = random.sample(range(len(self.TF_list)), self.TF_batch_size)
            Used_index = [self.protein_index_list[i] for i in samples]
            Used_embeddings = [self.embedding_list[i] for i in samples]

        # If you change the above part to use a HDF5 dataset of mean-embeddings
        # you can code it here so that Used_embeddings is a torch.tensor (C, H)
        # with C = number of TFs (classes) and H = hidden dimensions of the embeddings
        # then in your model, you only need to process it with linear layers.

        DNA = torch.stack([torch.tensor(sample[0]) for sample in batch])
        y = torch.stack([torch.tensor(sample[1]) for sample in batch])
        #DNA, y = torch.utils.data.default_collate(batch)
        y = y[:, Used_index]

        return DNA, torch.tensor(np.array(Used_embeddings)), y
    

class Collater_val2_0():
    def __init__(self, path, TF_list, val_list, embeddings, TF_batch_size = 0):
        f = h5torch.File(path)
        self.TF_batch_size = TF_batch_size
        self.TF_list = TF_list
        self.val_list = val_list
        self.protein_index_list = np.concatenate(
            [np.where(f["1/prots"][:].astype('str') == i) for i in self.TF_list]
            ).flatten()
        embedding_list_temp = []
        for i in self.protein_index_list:
            embedding_list_temp.append(
                f[embeddings][str(i)][:]
                )
        self.embedding_list = embedding_list_temp[:]
        
        self.protein_index_list_val = np.concatenate(
            [np.where(f["1/prots"][:].astype('str') == i) for i in self.val_list]
            ).flatten()
        embedding_list_temp = []
        for i in self.protein_index_list_val:
            embedding_list_temp.append(
                f[embeddings][str(i)][:]
                )
        self.embedding_list_val = embedding_list_temp[:]

    def __call__(self, batch):
        # SAMPLE TFs (Too many cause memory issues)
        Used_index = []
        Used_embeddings = []
        if self.TF_batch_size == 0:
            Used_index = self.protein_index_list
            Used_embeddings = self.embedding_list
        else:
            samples = random.sample(range(len(self.TF_list)), self.TF_batch_size)
            Used_index = [self.protein_index_list[i] for i in samples]
            Used_embeddings = [self.embedding_list[i] for i in samples]

        # If you change the above part to use a HDF5 dataset of mean-embeddings
        # you can code it here so that Used_embeddings is a torch.tensor (C, H)
        # with C = number of TFs (classes) and H = hidden dimensions of the embeddings
        # then in your model, you only need to process it with linear layers.

        DNA = torch.stack([torch.tensor(sample[0]) for sample in batch])
        y = torch.stack([torch.tensor(sample[1]) for sample in batch])
        #DNA, y = torch.utils.data.default_collate(batch)
        y_train = y[:, Used_index]
        
        ## same for val:
        Used_index_val = []
        Used_embeddings_val = []
        if self.TF_batch_size == 0:
            Used_index_val = self.protein_index_list_val[:]
            Used_embeddings_val = self.embedding_list_val[:]
        else:
            samples = random.sample(range(len(self.val_list)), self.TF_batch_size)
            Used_index_val = [self.protein_index_list_val[i] for i in samples]
            Used_embeddings_val = [self.embedding_list_val[i] for i in samples]

        y_val = y[:, Used_index_val]
        
        return DNA, torch.tensor(np.array(Used_embeddings)), y_train, torch.tensor(np.array(Used_embeddings_val)), y_val