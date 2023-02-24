import numpy as np
import h5torch
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from h5torch.dataset import apply_dtype
import random

class RemapDataset(h5torch.Dataset):
    def __init__(
        self,
        path,
        indices_to_sample,
        window_size=1024,
        resolution=128
    ):
        self.f = h5torch.File(path)
        self.indices = indices_to_sample
        self.window_size = window_size
        self.resolution = resolution
        self.start_position_offset = (self.window_size//2)-(self.resolution//2)
        self.stop_position_offset = (self.window_size//2)+(self.resolution//2)

        self.n_proteins = self.f["unstructured/protein_map"].shape[0]

        # self.debug_val = 0
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        
        start_ix = self.indices[index]
        end_ix = start_ix + self.window_size

        DNA = torch.tensor(
            apply_dtype(self.f["central"], self.f["central"][start_ix:end_ix])
            )

        slice_ix1, slice_ix2 = self.f["unstructured/slice_indices_per_100000"][:][[start_ix // 100_000, end_ix // 100_000], [0, 1]]
        peaks_start_sliced = self.f["unstructured/peaks_start"][slice_ix1:slice_ix2]
        peaks_end_sliced = self.f["unstructured/peaks_end"][slice_ix1:slice_ix2]
        peaks_prot_sliced = self.f["unstructured/peaks_prot"][slice_ix1:slice_ix2]

        indices = (peaks_start_sliced < end_ix) & (peaks_end_sliced > start_ix)

        labels = np.zeros(
            (self.n_proteins, self.window_size),
            dtype="int64"
            )
        if indices.sum() > 0:  # there has to be at least one peak in the sampled region, otherwise don't fill anything in
            start_peaks = np.maximum(peaks_start_sliced[indices] - start_ix, 0)
            end_peaks = np.minimum(peaks_end_sliced[indices] - start_ix, self.window_size)
            slices_ = [np.arange(srt, stp) for srt, stp in zip(start_peaks, end_peaks)]
            slice_lens = np.array([s.shape[0] for s in slices_])
            slices_ = np.array([np.pad(s, (0, lens_), mode="edge") for s, lens_ in zip(slices_, max(slice_lens) - slice_lens)])

            labels[peaks_prot_sliced[indices][:, None], slices_] = 1

        y = torch.tensor(labels)
        y = y[:, self.start_position_offset:self.stop_position_offset]

        # integrate labels over positions: why?
        y = (y.sum(-1) != 0).type(torch.float32)

        # the strategy below gives you a label per window_step nucleotides: "resolution", why not use this?:

        # https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
        # if (self.label_bin_window_size > 1) and (self.label_bin_window_step > 1):
        #     y = (y.unfold(
        #         1,
        #         size = self.label_bin_window_size,
        #         step = self.label_bin_window_step
        #         ).sum(-1) != 0).to(y)

        return DNA, y
    
class ReMapDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_loc,
        val_loc,
        test_loc,
        TF_list,
        TF_batch_size=0,
        window_size=1024,
        resolution_factor=128,
        batch_size: int = 32,
        embeddings="unstructured/prot_embeddings",
        num_workers = 3
    ):
        super().__init__()
        self.train_loc = train_loc
        self.val_loc = val_loc
        self.test_loc = test_loc
        self.batch_size = batch_size
        self.window_size = window_size
        self.resolution_factor = resolution_factor
        self.TF_list = TF_list
        self.TF_batch_size = TF_batch_size
        self.embeddings = embeddings
        self.num_workers = num_workers

        # TRAIN DATA
        ## GET INDICES
        f = h5torch.File(self.train_loc)
        indices = np.array([0] + list(np.cumsum(f["unstructured/chrom_lens"][:] + 10000)))
        start_pos_each_chrom = indices[:-1]
        end_pos_each_chrom = indices[1:] - 10000
        positions_to_sample = []
        for starts, stops in zip(start_pos_each_chrom, end_pos_each_chrom):
            positions_to_sample.append(np.arange(starts, stops, self.resolution_factor)[:-10]) ###!! THIS -10 WAS A QUICK FIX?? WHAT DOES IT DO, WHY IS IT NEEDED?
        self.positions_to_sample_train = np.concatenate(positions_to_sample)
        
        ## CREATE DATASET
        self.train_data = RemapDataset(
            self.train_loc,
            indices_to_sample=self.positions_to_sample_train,
            window_size=window_size,
            resolution=resolution_factor,
            )
        
        # VALIDATION DATA
        ## GET INDICES
        f = h5torch.File(self.val_loc)
        indices = np.array([0] + list(np.cumsum(f["unstructured/chrom_lens"][:] + 10000)))
        start_pos_each_chrom = indices[:-1]
        end_pos_each_chrom = indices[1:] - 10000
        positions_to_sample = []
        for starts, stops in zip(start_pos_each_chrom, end_pos_each_chrom):
            positions_to_sample.append(np.arange(starts, stops, self.resolution_factor)[:-10]) ###!! THIS -10 WAS A QUICK FIX?? WHAT DOES IT DO, WHY IS IT NEEDED?
        self.positions_to_sample_val = np.concatenate(positions_to_sample)
        
        ## CREATE DATASET
        self.val_data = RemapDataset(
            self.val_loc,
            indices_to_sample=self.positions_to_sample_val,
            window_size=window_size,
            resolution=resolution_factor,
            )
        
        # TEST DATA
        ## GET INDICES
        f = h5torch.File(self.test_loc)
        indices = np.array([0] + list(np.cumsum(f["unstructured/chrom_lens"][:] + 10000)))
        start_pos_each_chrom = indices[:-1]
        end_pos_each_chrom = indices[1:] - 10000
        positions_to_sample = []
        for starts, stops in zip(start_pos_each_chrom, end_pos_each_chrom):
            positions_to_sample.append(np.arange(starts, stops, self.resolution_factor)[:-10]) ###!! THIS -10 WAS A QUICK FIX?? WHAT DOES IT DO, WHY IS IT NEEDED?
        self.positions_to_sample_test = np.concatenate(positions_to_sample)
        
        ## CREATE DATASET
        self.test_data = RemapDataset(
            self.test_loc,
            indices_to_sample=self.positions_to_sample_test,
            window_size=window_size,
            resolution=resolution_factor,
            )


    def setup(self, stage=None):
        pass
            
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle = True,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=Collater(self.train_loc, self.TF_list, self.embeddings, TF_batch_size = self.TF_batch_size)
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=Collater(self.val_loc, self.TF_list, self.embeddings, TF_batch_size = self.TF_batch_size)
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, 
            batch_size=self.batch_size, 
            shuffle=False, 
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=Collater(self.test_loc, self.TF_list, self.embeddings, TF_batch_size = self.TF_batch_size)
        )


    # how is the predict dataloader different from train/val/test // does this need to be here?
    def predict_setup(self, Predict_TF, Data_split):
        if Data_split.lower() == "train":
            self.pred_loc = self.train_loc
            self.positions_to_sample_pred = self.positions_to_sample_train
        elif Data_split.lower() == "val":
            self.pred_loc = self.val_loc
            self.positions_to_sample_pred = self.positions_to_sample_val
        elif Data_split.lower() == "test":
            self.pred_loc = self.test_loc
            self.positions_to_sample_pred = self.positions_to_sample_test
        else:
            raise Exception("Not a valid data split")
        self.predict_data = RemapDataset(
            self.pred_loc,
            TF_list=Predict_TF,
            TF_batch_size=0,
            indices_to_sample=self.positions_to_sample_pred,
            window_size=self.window_size,
            resolution=self.resolution_factor,
            embeddings=self.embeddings
            )
    
    def predict_dataloader(self):
        return DataLoader(self.predict_data, batch_size=self.batch_size)



class Collater():
    def __init__(self, path, TF_list, embeddings, TF_batch_size = 0):
        f = h5torch.File(path)
        self.TF_batch_size = TF_batch_size
        self.TF_list = TF_list
        self.protein_index_list = np.concatenate(
            [np.where(f["unstructured/protein_map"][:].astype('str') == i) for i in self.TF_list]
            ).flatten()
        embedding_list_temp = []
        for i in self.protein_index_list:
            embedding_list_temp.append(
                #self.f["unstructured/prot_embeddings"][str(i)][:]
                f[embeddings][str(i)][:]
                )
        self.embedding_list = embedding_list_temp

    def __call__(self, batch):
        # SAMPLE TFs (Too many cause memory issues)
        Used_index = []
        Used_embeddings = []
        if self.TF_batch_size == 0:
            Used_index = self.protein_index_list[:]
            Used_embeddings = self.embedding_list[:]
        else:
            samples = random.sample(range(len(self.TF_list)), self.TF_batch_size)
            Used_index = [self.protein_index_list[i] for i in samples]
            Used_embeddings = [self.embedding_list[i] for i in samples]

        # If you change the above part to use a HDF5 dataset of mean-embeddings
        # you can code it here so that Used_embeddings is a torch.tensor (C, H)
        # with C = number of TFs (classes) and H = hidden dimensions of the embeddings
        # then in your model, you only need to process it with linear layers.

        DNA, y = torch.utils.data.default_collate(batch)
        y = y[:, Used_index]

        return DNA, Used_embeddings, y