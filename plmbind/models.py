import numpy as np
import h5torch
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.classification import MultilabelF1Score
from torchmetrics.classification import MultilabelAUROC
from h5torch.dataset import apply_dtype

class FullTFModel(pl.LightningModule):
    """
    Still have to add this.
    """
    def __init__(
        self,
        seq_len,
        prot_embedding_dim,
        TF_list,
        num_classes,
        num_DNA_filters=20,
        num_prot_filters=20,
        DNA_kernel_size=10,
        prot_kernel_size=10,
        dropout=0.25,
        num_linear_layers=3,
        linear_layer_size=64,
        linear_layer_size_prot = 64,
        final_embeddings_size=64
    ):
        super(FullTFModel, self).__init__()

        self.TF_list = TF_list
        # Define metrics and loss function
        self.loss_function = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy(compute_on_step=False)
        self.val_acc = BinaryAccuracy(compute_on_step=False)
        self.test_acc = BinaryAccuracy(compute_on_step=False)

        self.train_AUROC = MultilabelAUROC(
            num_labels=num_classes,
            average='macro',
            compute_on_step=False
            )
        self.train_AUROC_micro = MultilabelAUROC(
            num_labels=num_classes,
            average='micro',
            compute_on_step=False
            )
        self.val_AUROC = MultilabelAUROC(
            num_labels=num_classes,
            average='macro',
            compute_on_step=False
            )
        self.val_AUROC_micro = MultilabelAUROC(
            num_labels=num_classes,
            average='micro',
            compute_on_step=False
            )
        self.val_AUROC_all = MultilabelAUROC(
            num_labels=num_classes,
            average='none',#'macro',
            compute_on_step=False
            )
        self.test_AUROC = MultilabelAUROC(
            num_labels=num_classes,
            average='macro',
            compute_on_step=False
            )
        self.test_AUROC_micro = MultilabelAUROC(
            num_labels=num_classes,
            average='micro',
            compute_on_step=False
            )
        self.seq_len = seq_len
        nucleotide_weights = torch.FloatTensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0]]
            )
        self.embedding = nn.Embedding.from_pretrained(nucleotide_weights)

        # The DNA branch of the model
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, num_DNA_filters, kernel_size=DNA_kernel_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                num_DNA_filters*(seq_len-DNA_kernel_size+1),
                linear_layer_size
                ),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, final_embeddings_size)
        )
        
        # The Protein branch of the model
        self.conv_net_proteins = nn.Sequential(
            #nn.AdaptiveMaxPool1d(AdaptiveMaxPoolingOutput),
            nn.Conv1d(prot_embedding_dim, num_prot_filters, prot_kernel_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                num_prot_filters*(1024-prot_kernel_size+1), #num_prot_filters*(AdaptiveMaxPoolingOutput-prot_kernel_size+1),
                linear_layer_size_prot
                ),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size_prot, final_embeddings_size)
        )
        self.save_hyperparameters()

    def forward(self, x_DNA_in, x_prot_in):
        # Run DNA branch
        x_DNA = self.conv_net(self.embedding(x_DNA_in).permute(0, 2, 1))

        # Run protein branch for every protein in the input.
        # This might not be the most efficient way of doing this!!
        x_prot = self.conv_net_proteins(x_prot_in.permute(0, 2, 1))
    

        # Take the dot product
        x_product = torch.einsum("b l, h l -> h b", x_prot, x_DNA) ## dubble check ##!! !!!!!!!!!!!!!!!!!!

        return x_product

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_DNA, x_prot, y = train_batch
        y_hat = self(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss)
        
        y_hat_sigmoid = torch.sigmoid(y_hat)
        self.train_acc(y_hat_sigmoid, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, batch_size=1000)
        self.train_AUROC(y_hat_sigmoid, y)
        self.log('train_AUROC', self.train_AUROC, on_step=False, on_epoch=True, batch_size=1000)
        self.train_AUROC_micro(y_hat_sigmoid, y)
        self.log('train_AUROC_micro', self.train_AUROC_micro, on_step=False, on_epoch=True, batch_size=1000)
        return loss

    def test_step(self, test_batch, batch_idx):
        x_DNA, x_prot, y = test_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y)
        self.log('test_loss', loss)
        
        y_hat_sigmoid = torch.sigmoid(y_hat)
        self.test_acc(y_hat_sigmoid, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, batch_size=1000)
        self.test_AUROC(y_hat_sigmoid, y)
        self.log('test_AUROC', self.test_AUROC, on_step=False, on_epoch=True, batch_size=1000)
        self.test_AUROC_micro(y_hat_sigmoid, y)
        self.log('test_AUROC_micro', self.test_AUROC_micro, on_step=False, on_epoch=True, batch_size=1000)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_DNA, x_prot, y = valid_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y)
        self.log('val_loss', loss)
        
        y_hat_sigmoid = torch.sigmoid(y_hat)
        self.val_acc(y_hat_sigmoid, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, batch_size=1000)
    
        self.val_AUROC(y_hat_sigmoid, y)
        self.log('val_AUROC', self.val_AUROC, on_step=False, on_epoch=True, batch_size=1000)
        
        self.val_AUROC_micro(y_hat_sigmoid, y)
        self.log('val_AUROC_micro', self.val_AUROC_micro, on_step=False, on_epoch=True, batch_size=1000)
        
        all_AUROC = self.val_AUROC_all(y_hat_sigmoid,y)
        all_AUROC_dict = dict([(key, value) for key, value in zip(self.TF_list, all_AUROC)])
        self.log_dict(all_AUROC_dict, on_step=False, on_epoch=True, batch_size=1000)
        return loss

    def predict_step(self, batch, batch_idx):
        x_DNA, x_prot, y = batch
        y_hat = self.forward(x_DNA, x_prot)
        y_hat_sigmoid = torch.sigmoid(y_hat)
        return y, y_hat, y_hat_sigmoid
    
    def get_TF_latent_vector(self, TF_emb):
        return self.conv_net_proteins(TF_emb.permute(0, 2, 1))
        

    
class MultilabelModel(pl.LightningModule):
    """
    Still have to add this.
    """
    def __init__(
        self,
        seq_len,
        num_classes,
        num_DNA_filters=20,
        DNA_kernel_size=10,
        dropout=0.25,
        num_linear_layers=3,
        linear_layer_size=64
    ):
        super(MultilabelModel, self).__init__()

        # Define metrics and loss function
        self.loss_function = nn.BCEWithLogitsLoss()
        self.train_acc = BinaryAccuracy(compute_on_step=False)
        self.val_acc = BinaryAccuracy(compute_on_step=False)
        self.test_acc = BinaryAccuracy(compute_on_step=False)

        self.train_AUROC = MultilabelAUROC(
            num_labels=num_classes,
            average='macro',
            compute_on_step=False
            )
        self.val_AUROC = MultilabelAUROC(
            num_labels=num_classes,
            average='macro',
            compute_on_step=False
            )
        self.test_AUROC = MultilabelAUROC(
            num_labels=num_classes,
            average='macro',
            compute_on_step=False
            )
        self.seq_len = seq_len
        nucleotide_weights = torch.FloatTensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0]]
            )
        self.embedding = nn.Embedding.from_pretrained(nucleotide_weights)

        # The DNA branch of the model
        self.conv_net = nn.Sequential(
            nn.Conv1d(4, num_DNA_filters, kernel_size=DNA_kernel_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(
                num_DNA_filters*(seq_len-DNA_kernel_size+1),
                linear_layer_size
                ),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, linear_layer_size),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_layer_size, num_classes)
        )
        
        self.save_hyperparameters()

    def forward(self, x_DNA_in, x_prot_in):
        # Run DNA branch
        x_DNA = self.conv_net(self.embedding(x_DNA_in).permute(0, 2, 1))

        return x_DNA

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_DNA, x_prot, y = train_batch
        y_hat = self(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss)
        y_hat_sigmoid = torch.sigmoid(y_hat)
        self.train_acc(y_hat_sigmoid, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, batch_size=1000)
        self.train_AUROC(y_hat_sigmoid, y)
        self.log('train_AUROC', self.train_AUROC, on_step=False, on_epoch=True, batch_size=1000)
        return loss

    def test_step(self, test_batch, batch_idx):
        x_DNA, x_prot, y = test_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y)
        self.log('test_loss', loss)
        
        y_hat_sigmoid = torch.sigmoid(y_hat)
        self.test_acc(y_hat_sigmoid, y)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, batch_size=1000)
        self.test_AUROC(y_hat_sigmoid, y)
        self.log('test_AUROC', self.test_AUROC, on_step=False, on_epoch=True, batch_size=1000)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_DNA, x_prot, y = valid_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y)
        self.log('val_loss', loss)
        
        y_hat_sigmoid = torch.sigmoid(y_hat)
        self.val_acc(y_hat_sigmoid, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, batch_size=1000)
    
        self.val_AUROC(y_hat_sigmoid, y)
        self.log('val_AUROC', self.val_AUROC, on_step=False, on_epoch=True, batch_size=1000)
        return loss

    def predict_step(self, batch, batch_idx):
        x_DNA, x_prot, y = batch
        y_hat = self.forward(x_DNA, x_prot)
        y_hat_sigmoid = torch.sigmoid(y_hat)
        return y, y_hat, y_hat_sigmoid