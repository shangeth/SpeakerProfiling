import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics.regression import MeanAbsoluteError as MAE
from pytorch_lightning.metrics.regression import MeanSquaredError  as MSE
from pytorch_lightning.metrics.classification import Accuracy

import pandas as pd
import wavencoder
import torch_optimizer as optim


from Model.models import Wav2VecLSTMH, SpectralLSTMH, MultiScaleH
from Model.utils import RMSELoss

class LightningModel(pl.LightningModule):
    def __init__(self, HPARAMS):
        super().__init__()
        # HPARAMS
        self.save_hyperparameters()
        self.models = {
            'wav2vecLSTMAttn': Wav2VecLSTMH,
            'MultiScale' : MultiScaleH,
            'LSTMAttn' : SpectralLSTMH,
        }
        self.model = self.models[HPARAMS['model_type']](HPARAMS['hidden_size'])

        self.regression_criterion = MSE()
        self.mae_criterion = MAE()
        self.rmse_criterion = RMSELoss()

        self.lr = HPARAMS['lr']

        self.csv_path = HPARAMS['speaker_csv_path']
        self.df = pd.read_csv(self.csv_path)
        self.h_mean = self.df['height'].mean()
        self.h_std = self.df['height'].std()

        print(f"Model Details: #Params = {self.count_total_parameters()}\t#Trainable Params = {self.count_trainable_parameters()}")

    def count_total_parameters(self):
            return sum(p.numel() for p in self.parameters())
    
    def count_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]

    def training_step(self, batch, batch_idx):
        x, y_h, y_a, y_g = batch
        y_hat_h = self(x)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h = y_hat_h.view(-1).float()

        height_loss = self.regression_criterion(y_hat_h, y_h)
        loss = height_loss

        height_mae = self.mae_criterion(y_hat_h*self.h_std+self.h_mean, y_h*self.h_std+self.h_mean)

        return {'loss':loss, 
                'train_height_mae':height_mae.item(),
                }
    
    def training_epoch_end(self, outputs):
        n_batch = len(outputs)
        loss = torch.tensor([x['loss'] for x in outputs]).mean()
        height_mae = torch.tensor([x['train_height_mae'] for x in outputs]).sum()/n_batch

        self.log('train/loss' , loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/h',height_mae.item(), on_step=False, on_epoch=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, y_h, y_a, y_g = batch
        y_hat_h = self(x)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h = y_hat_h.view(-1).float()

        height_loss = self.regression_criterion(y_hat_h, y_h)
        loss = height_loss

        height_mae = self.mae_criterion(y_hat_h*self.h_std+self.h_mean, y_h*self.h_std+self.h_mean)

        return {'val_loss':loss, 
                'val_height_mae':height_mae.item(),
                }

    def validation_epoch_end(self, outputs):
        n_batch = len(outputs)
        val_loss = torch.tensor([x['val_loss'] for x in outputs]).mean()
        height_mae = torch.tensor([x['val_height_mae'] for x in outputs]).sum()/n_batch
        
        self.log('val/loss' , val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/h',height_mae.item(), on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y_h, y_a, y_g = batch
        y_hat_h = self(x)
        y_h, y_a, y_g = y_h.view(-1).float(), y_a.view(-1).float(), y_g.view(-1).float()
        y_hat_h = y_hat_h.view(-1).float()

        idx = y_g.view(-1).long()
        female_idx = torch.nonzero(idx).view(-1)
        male_idx = torch.nonzero(1-idx).view(-1)

        male_height_mae = self.mae_criterion(y_hat_h[male_idx]*self.h_std+self.h_mean, y_h[male_idx]*self.h_std+self.h_mean)
        femal_height_mae = self.mae_criterion(y_hat_h[female_idx]*self.h_std+self.h_mean, y_h[female_idx]*self.h_std+self.h_mean)
        male_height_rmse = self.rmse_criterion(y_hat_h[male_idx]*self.h_std+self.h_mean, y_h[male_idx]*self.h_std+self.h_mean)
        femal_height_rmse = self.rmse_criterion(y_hat_h[female_idx]*self.h_std+self.h_mean, y_h[female_idx]*self.h_std+self.h_mean)
        return {
                'male_height_mae':male_height_mae.item(),
                'female_height_mae':femal_height_mae.item(),
                'male_height_rmse':male_height_rmse.item(),
                'femal_height_rmse':femal_height_rmse.item()}

    def test_epoch_end(self, outputs):
        n_batch = len(outputs)
        male_height_mae = torch.tensor([x['male_height_mae'] for x in outputs]).mean()
        female_height_mae = torch.tensor([x['female_height_mae'] for x in outputs]).mean()

        male_height_rmse = torch.tensor([x['male_height_rmse'] for x in outputs]).mean()
        femal_height_rmse = torch.tensor([x['femal_height_rmse'] for x in outputs]).mean()

        pbar = {'male_height_mae' : male_height_mae.item(),
                'female_height_mae':female_height_mae.item(),
                'male_height_rmse' : male_height_rmse.item(),
                'femal_height_rmse':femal_height_rmse.item()}
        self.logger.log_hyperparams(pbar)
        self.log_dict(pbar)