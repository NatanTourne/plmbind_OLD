from ast import Global
from faulthandler import cancel_dump_traceback_later
import numpy as np
import h5torch
import torch
import pytorch_lightning as pl
from torch import optim, nn
from transformers import EsmModel, EsmTokenizer

class Crop(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n
    def forward(self, x):
        return x[..., self.n:-self.n]
    
class Permute(nn.Module): 
    def __init__(self, *args):
        super().__init__()
        self.args = args
    def forward(self, x):
        return x.permute(*self.args)
    
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim = 64, kernel_size = 5, dropout = 0.2): # a simple residual block
        super().__init__()
        self.net = nn.Sequential(
            Permute(0,2,1),
            nn.LayerNorm(hidden_dim),
            Permute(0,2,1),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding = "same"), #channels in, channels out, kernel size
            nn.ReLU(),
            nn.Dropout(dropout)
            )
        # NOTE: these permutes are necessary here because LayerNorm expects the "hidden_dim" to be the last dim
        # While for Conv1d the "channels" or "hidden_dims" are the second dimension.
        # These permutes basically swap BxCxL to BxLxC for the layernorm, and afterwards swap them back
    def forward(self, x):
        return self.net(x) + x #residual connection
    
class GlobalPool(nn.Module):
    def __init__(self, pooled_axis = 1, mode = "max"):
        super().__init__()
        assert mode in ["max", "mean"], "Only max and mean-pooling are implemented"
        if mode == "max":
            self.op = lambda x: torch.max(x, axis = pooled_axis).values
        elif mode == "mean":
            self.op = lambda x: torch.mean(x, axis = pooled_axis).values
    def forward(self, x):
        return self.op(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, y_hat, y):
        
        loss = -(y.float()*(1-y_hat.float().sigmoid())**(self.gamma)*y_hat.float().sigmoid().log()+(1-y.float())*(y_hat.float().sigmoid())**(self.gamma)*(1-y_hat.float().sigmoid()).log())
        return loss.mean()
    
class filteredBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = nn.BCEWithLogitsLoss()
    def forward(self, y_hat, y):
       
        return self.loss_function(y_hat.permute(1,0,2)[:,y.sum(dim=1)!=0], y.permute(1,0,2)[:,y.sum(dim=1)!=0].float())

class DNA_branch(nn.Module):
    def __init__(self, num_DNA_filters, DNA_kernel_size, DNA_dropout):
        super().__init__()
        self.conv_net = nn.Sequential(
                nn.Conv1d(4, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2)
        )
    def forward(self, x):
        return self.conv_net(x)

class DNA_branch_contrastive(nn.Module):
    def __init__(self, num_DNA_filters, DNA_kernel_size, DNA_dropout):
        super().__init__()
        self.conv_net = nn.Sequential(
                nn.Conv1d(4, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(DNA_dropout),
                
                ResidualBlock(num_DNA_filters, DNA_kernel_size, DNA_dropout),
                
                Permute(0,2,1),
                nn.LayerNorm(num_DNA_filters),
                Permute(0,2,1),
                nn.Conv1d(num_DNA_filters, num_DNA_filters, kernel_size=DNA_kernel_size, padding="same"),
                nn.ReLU(),
                nn.GlobalPool()
        )
    def forward(self, x):
        return self.conv_net(x)
    
class PlmbindContrastive(pl.LightningDataModule):
    def __init__(
        self,
        seq_len,
        prot_embedding_dim,
        num_DNA_filters=20,
        num_prot_filters=20,
        DNA_kernel_size=10,
        prot_kernel_size=10,
        initial_prot_kernel_size=1,
        DNA_dropout=0.25,
        protein_dropout=0.25,
        linear_layer_size_prot = 64,
        final_embeddings_size = 64,
        learning_rate_protein_branch=1e-5,
        learning_rate_DNA_branch=1e-5,
        calculate_val_tf_loss = False,
        DNA_branch_path = "None",
        loss_function = "BCE",
        loss_weights = None,
        gamma = 2
        
    ):
        super(PlmbindContrastive, self).__init__()
        self.learning_rate_protein_branch = learning_rate_protein_branch
        self.learning_rate_DNA_branch = learning_rate_DNA_branch
        self.calculate_val_tf_loss = calculate_val_tf_loss
        self.gamma = gamma
        # Define metrics and loss function
        if loss_function == "BCE":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif loss_function =="weighted_BCE":
            self.loss_weights = loss_weights
            self.loss_function = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(loss_weights))
        elif loss_function == "focal":
            self.loss_function = FocalLoss(self.gamma) # GAAT NOG ERROR GEVEN DOOR DAT FLOAT 
        elif loss_function == "filtered_BCE":
            self.loss_function = filteredBCE()
            
        self.seq_len = seq_len
        nucleotide_weights = torch.FloatTensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0]]
            )
        self.embedding = nn.Embedding.from_pretrained(nucleotide_weights)
        self.DNA_branch = DNA_branch_contrastive(num_DNA_filters, DNA_kernel_size, DNA_dropout)
        self.conv_net_proteins = nn.Sequential(
            nn.Conv1d(prot_embedding_dim, int(prot_embedding_dim/4), kernel_size = initial_prot_kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(protein_dropout),
            
            Permute(0,2,1),
            nn.LayerNorm(int(prot_embedding_dim/4)),
            Permute(0,2,1),
            nn.Conv1d(int(prot_embedding_dim/4), int(prot_embedding_dim/8), initial_prot_kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(protein_dropout),
            
            Permute(0,2,1),
            nn.LayerNorm(int(prot_embedding_dim/8)),
            Permute(0,2,1),
            nn.Conv1d(int(prot_embedding_dim/8), num_prot_filters, initial_prot_kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(protein_dropout),
            
            ResidualBlock(num_prot_filters, prot_kernel_size, protein_dropout),
            ResidualBlock(num_prot_filters, prot_kernel_size, protein_dropout),
            ResidualBlock(num_prot_filters, prot_kernel_size, protein_dropout),

            GlobalPool(pooled_axis = 2, mode = 'max'),
            nn.Flatten(),
            
            nn.LayerNorm(num_prot_filters), # before or after linear layer??
            nn.Linear(num_prot_filters,linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, final_embeddings_size)
        )
        self.save_hyperparameters()
    def forward(self, x_DNA_in, x_prot_in):
        x_DNA = self.DNA_branch(self.embedding(x_DNA_in).permute(0, 2, 1))

        x_prot = self.conv_net_proteins(x_prot_in.permute(0, 2, 1)) 
    
        x_product = torch.einsum("b h l, c h -> b c l", x_DNA, x_prot)

        return x_product
    
    def configure_optimizers(self):
        optimizer = optim.Adam([
            {"params":self.DNA_branch.parameters(), "lr":self.learning_rate_DNA_branch},
            {"params":self.conv_net_proteins.parameters(), "lr":self.learning_rate_protein_branch}
            ])
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        x_DNA, x_prot, y = train_batch
        y_hat = self(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x_DNA, x_prot, y = test_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y.float())
        self.log('test_loss', loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_DNA, x_prot, y, x_prot_val, y_val = valid_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y.float())
        self.log('val_loss_DNA', loss)
        
        if self.calculate_val_tf_loss:
            y_hat_val = self.forward(x_DNA, x_prot_val)
            loss_val = self.loss_function(y_hat_val, y_val.float())
            self.log('val_loss_TF', loss_val)

        return loss
        
    def predict_step(self, batch, batch_idx):
        x_DNA, x_prot, y = batch
        y_hat = self.forward(x_DNA, x_prot)
        y_hat_sigmoid = torch.sigmoid(y_hat).detach()
        return y.to(torch.int8), y_hat.to(torch.float16), y_hat_sigmoid.to(torch.float16)
        #return y, y_hat, y_hat_sigmoid
    
    def get_TF_latent_vector(self, TF_emb):
        return self.conv_net_proteins(TF_emb.permute(0, 2, 1))

class PlmbindFullModel(pl.LightningModule):
    """
    Implement:
        - Weighting the loss function: NOT TESTED YET
        - focal loss: NOT TESTED YET
        - different protein branches??
    """
    def __init__(
        self,
        seq_len,
        prot_embedding_dim,
        num_DNA_filters=20,
        num_prot_filters=20,
        DNA_kernel_size=10,
        prot_kernel_size=10,
        initial_prot_kernel_size=1,
        DNA_dropout=0.25,
        protein_dropout=0.25,
        linear_layer_size_prot = 64,
        final_embeddings_size = 64,
        learning_rate_protein_branch=1e-5,
        learning_rate_DNA_branch=1e-5,
        calculate_val_tf_loss = True,
        DNA_branch_path = "None",
        loss_function = "BCE",
        loss_weights = None,
        gamma = 2
        
    ):
        super(PlmbindFullModel, self).__init__()
        self.learning_rate_protein_branch = learning_rate_protein_branch
        self.learning_rate_DNA_branch = learning_rate_DNA_branch
        self.calculate_val_tf_loss = calculate_val_tf_loss
        self.gamma = gamma
        # Define metrics and loss function
        if loss_function == "BCE":
            self.loss_function = nn.BCEWithLogitsLoss()
        elif loss_function =="weighted_BCE":
            self.loss_weights = loss_weights
            self.loss_function = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(loss_weights))
        elif loss_function == "focal":
            self.loss_function = FocalLoss(self.gamma) # GAAT NOG ERROR GEVEN DOOR DAT FLOAT 
        elif loss_function == "filtered_BCE":
            self.loss_function = filteredBCE()
            
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
        # if resolution is 128: this means that we need 7 times pooling layers that half the "nucleotide"
        # dimension. After that we also need to shave off the edges, which I set in the dataloader to 4
        # also, you need to use padding in the convs to keep the dimensions there constant.
        if DNA_branch_path == "None":
            print("Initializing new DNA branch")
            self.DNA_branch = DNA_branch(num_DNA_filters, DNA_kernel_size, DNA_dropout)
        else:
            print("Using pretrained DNA branch")
            self.DNA_branch = DNA_branch(num_DNA_filters, DNA_kernel_size, DNA_dropout)
            DNA_model_checkpoint = torch.load(DNA_branch_path)
            full_pretrained_dict = DNA_model_checkpoint["state_dict"]
            pretrained_dict = {k[11:]: v for k, v in full_pretrained_dict.items() if k.split(".")[0] == "DNA_branch"}            
            self.DNA_branch.load_state_dict(state_dict = pretrained_dict)
            
        self.conv_net_final_layer = nn.Sequential(nn.Conv1d(num_DNA_filters, final_embeddings_size, kernel_size=1))
        # The Protein branch of the model
        self.conv_net_proteins = nn.Sequential(
            nn.Conv1d(prot_embedding_dim, int(prot_embedding_dim/4), kernel_size = initial_prot_kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(protein_dropout),
            
            Permute(0,2,1),
            nn.LayerNorm(int(prot_embedding_dim/4)),
            Permute(0,2,1),
            nn.Conv1d(int(prot_embedding_dim/4), int(prot_embedding_dim/8), initial_prot_kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(protein_dropout),
            
            Permute(0,2,1),
            nn.LayerNorm(int(prot_embedding_dim/8)),
            Permute(0,2,1),
            nn.Conv1d(int(prot_embedding_dim/8), num_prot_filters, initial_prot_kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(protein_dropout),
            
            ResidualBlock(num_prot_filters, prot_kernel_size, protein_dropout),
            ResidualBlock(num_prot_filters, prot_kernel_size, protein_dropout),
            ResidualBlock(num_prot_filters, prot_kernel_size, protein_dropout),

            GlobalPool(pooled_axis = 2, mode = 'max'),
            nn.Flatten(),
            
            nn.LayerNorm(num_prot_filters), # before or after linear layer??
            nn.Linear(num_prot_filters,linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, final_embeddings_size)
        )
        self.save_hyperparameters()

    def forward(self, x_DNA_in, x_prot_in):
        # Run DNA branch
        x_DNA = self.conv_net_final_layer(self.DNA_branch(self.embedding(x_DNA_in).permute(0, 2, 1))) # B x H x L_y

        x_prot = self.conv_net_proteins(x_prot_in.permute(0, 2, 1)) # C x H
    
        # with `y` being B x C x L_y

        # Take the dot product
        x_product = torch.einsum("b h l, c h -> b c l", x_DNA, x_prot)

        return x_product

    def configure_optimizers(self):
        optimizer = optim.Adam([
            {"params":self.DNA_branch.parameters(), "lr":self.learning_rate_DNA_branch},
            {"params":self.conv_net_final_layer.parameters(), "lr":self.learning_rate_DNA_branch},
            {"params":self.conv_net_proteins.parameters(), "lr":self.learning_rate_protein_branch}
            ])
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_DNA, x_prot, y = train_batch
        y_hat = self(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x_DNA, x_prot, y = test_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y.float())
        self.log('test_loss', loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_DNA, x_prot, y, x_prot_val, y_val = valid_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y.float())
        self.log('val_loss_DNA', loss)
        
        if self.calculate_val_tf_loss:
            y_hat_val = self.forward(x_DNA, x_prot_val)
            loss_val = self.loss_function(y_hat_val, y_val.float())
            self.log('val_loss_TF', loss_val)

        return loss
        
    def predict_step(self, batch, batch_idx):
        x_DNA, x_prot, y = batch
        y_hat = self.forward(x_DNA, x_prot)
        y_hat_sigmoid = torch.sigmoid(y_hat).detach()
        return y.to(torch.int8), y_hat.to(torch.float16), y_hat_sigmoid.to(torch.float16)
        #return y, y_hat, y_hat_sigmoid
    
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
        DNA_dropout=0.25,
        learning_rate=0.00001
    ):
        super(MultilabelModel, self).__init__()

        # Define metrics and loss function
        self.loss_function = nn.BCEWithLogitsLoss()
        self.learning_rate=learning_rate
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
        self.DNA_branch = DNA_branch(num_DNA_filters, DNA_kernel_size, DNA_dropout)
        self.conv_net_final_layer = nn.Sequential(nn.Conv1d(num_DNA_filters, num_classes, kernel_size=1))
        self.save_hyperparameters()

    def forward(self, x_DNA_in):
        # Run DNA branch
        x_DNA = self.conv_net_final_layer(self.DNA_branch(self.embedding(x_DNA_in).permute(0, 2, 1)))
        return x_DNA

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_DNA, x_prot, y = train_batch
        y_hat = self(x_DNA)
        loss = self.loss_function(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x_DNA, x_prot, y = test_batch
        y_hat = self.forward(x_DNA)
        loss = self.loss_function(y_hat, y.float())
        self.log('test_loss', loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_DNA, x_prot, y, x_prot_val, y_val = valid_batch
        y_hat = self.forward(x_DNA)
        loss = self.loss_function(y_hat, y.float())
        self.log('val_loss', loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x_DNA, x_prot, y = batch
        y_hat = self.forward(x_DNA)
        y_hat_sigmoid = torch.sigmoid(y_hat).detach()
        return y.to(torch.int8), y_hat.to(torch.float16), y_hat_sigmoid.to(torch.float16)

    
class PlmbindFullModel_kmer(pl.LightningModule):
    """
    
    """
    def __init__(
        self,
        seq_len,
        num_kmers=8000,
        kmer_embedding_size=32,
        num_DNA_filters=20,
        num_prot_filters=20,
        DNA_kernel_size=10,
        prot_kernel_size=10,
        initial_prot_kernel_size=1,
        DNA_dropout=0.25,
        protein_dropout=0.25,
        linear_layer_size_prot = 64,
        final_embeddings_size=64,
        learning_rate=1e-5,
        calculate_val_tf_loss = True,
        DNA_branch_path = "None"
    ):
        super(PlmbindFullModel_kmer, self).__init__()
        self.learning_rate = learning_rate
        self.calculate_val_tf_loss = calculate_val_tf_loss
        # Define metrics and loss function
        self.loss_function = nn.BCEWithLogitsLoss()
        self.seq_len = seq_len
        nucleotide_weights = torch.FloatTensor(
            [[1, 0, 0, 0],
             [0, 1, 0, 0],
             [0, 0, 1, 0],
             [0, 0, 0, 1],
             [0, 0, 0, 0]]
            )
        self.embedding = nn.Embedding.from_pretrained(nucleotide_weights)
        if DNA_branch_path == "None":
            print("Initializing new DNA branch")
            self.DNA_branch = DNA_branch(num_DNA_filters, DNA_kernel_size, DNA_dropout)
        else:
            print("Using pretrained DNA branch")
            self.DNA_branch = DNA_branch(num_DNA_filters, DNA_kernel_size, DNA_dropout)
            DNA_model_checkpoint = torch.load(DNA_branch_path)
            full_pretrained_dict = DNA_model_checkpoint["state_dict"]
            pretrained_dict = {k[11:]: v for k, v in full_pretrained_dict.items() if k.split(".")[0] == "DNA_branch"}            
            self.DNA_branch.load_state_dict(state_dict = pretrained_dict)
        # The DNA branch of the model
        # if resolution is 128: this means that we need 7 times pooling layers that half the "nucleotide"
        # dimension. After that we also need to shave off the edges, which I set in the dataloader to 4
        # also, you need to use padding in the convs to keep the dimensions there constant.
        self.conv_net_final_layer = nn.Sequential(nn.Conv1d(num_DNA_filters, final_embeddings_size, kernel_size=1))
        # The Protein branch of the model
        self.conv_net_proteins = nn.Sequential(
            nn.Embedding(num_embeddings=num_kmers, embedding_dim=kmer_embedding_size, padding_idx=0),
            Permute(0,2,1),
            nn.Conv1d(kmer_embedding_size, int(kmer_embedding_size/4), kernel_size = initial_prot_kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(protein_dropout),
            
            Permute(0,2,1),
            nn.LayerNorm(int(kmer_embedding_size/4)),
            Permute(0,2,1),
            nn.Conv1d(int(kmer_embedding_size/4), int(kmer_embedding_size/8), initial_prot_kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(protein_dropout),
            
            Permute(0,2,1),
            nn.LayerNorm(int(kmer_embedding_size/8)),
            Permute(0,2,1),
            nn.Conv1d(int(kmer_embedding_size/8), num_prot_filters, initial_prot_kernel_size, padding="same"),
            nn.ReLU(),
            nn.Dropout(protein_dropout),
            
            ResidualBlock(num_prot_filters, prot_kernel_size, protein_dropout),
            ResidualBlock(num_prot_filters, prot_kernel_size, protein_dropout),
            ResidualBlock(num_prot_filters, prot_kernel_size, protein_dropout),

            GlobalPool(pooled_axis = 2, mode = 'max'),
            nn.Flatten(), 
            
            nn.LayerNorm(num_prot_filters), # before or after linear layer??
            nn.Linear(num_prot_filters,linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, linear_layer_size_prot),
            nn.Dropout(protein_dropout),
            nn.ReLU(),
            nn.LayerNorm(linear_layer_size_prot),
            nn.Linear(linear_layer_size_prot, final_embeddings_size)
        )
        self.save_hyperparameters()

    def forward(self, x_DNA_in, x_prot_in):
        # Run DNA branch
        x_DNA = self.conv_net_final_layer(self.DNA_branch(self.embedding(x_DNA_in).permute(0, 2, 1))) # B x H x L_y

        x_prot = self.conv_net_proteins(x_prot_in) # C x H
    
        # with `y` being B x C x L_y

        # Take the dot product
        x_product = torch.einsum("b h l, c h -> b c l", x_DNA, x_prot)

        return x_product

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x_DNA, x_prot, y = train_batch
        y_hat = self(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y.float())
        self.log('train_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x_DNA, x_prot, y = test_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y.float())
        self.log('test_loss', loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        x_DNA, x_prot, y, x_prot_val, y_val = valid_batch
        y_hat = self.forward(x_DNA, x_prot)
        loss = self.loss_function(y_hat, y.float())
        self.log('val_loss_DNA', loss)
        
        if self.calculate_val_tf_loss:
            y_hat_val = self.forward(x_DNA, x_prot_val)
            loss_val = self.loss_function(y_hat_val, y_val.float())
            self.log('val_loss_TF', loss_val)

        return loss
        
    def predict_step(self, batch, batch_idx):
        x_DNA, x_prot, y = batch
        y_hat = self.forward(x_DNA, x_prot)
        y_hat_sigmoid = torch.sigmoid(y_hat).detach()
        return y.to(torch.int8), y_hat.to(torch.float16), y_hat_sigmoid.to(torch.float16)
        #return y, y_hat, y_hat_sigmoid
    
    def get_TF_latent_vector(self, TF_emb):
        return self.conv_net_proteins(TF_emb)
    