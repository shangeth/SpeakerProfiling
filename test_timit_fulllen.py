from argparse import ArgumentParser
from multiprocessing import Pool
import os

from TIMIT.dataset import TIMITDataset
from TIMIT.lightning_model import LightningModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
import pytorch_lightning as pl
from config import TIMITConfig

import torch
import torch.utils.data as data
# torch.use_deterministic_algorithms(True)

from tqdm import tqdm 
import pandas as pd
import numpy as np

if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=TIMITConfig.data_path)
    parser.add_argument('--speaker_csv_path', type=str, default=TIMITConfig.speaker_csv_path)
    parser.add_argument('--timit_wav_len', type=int, default=TIMITConfig.timit_wav_len)
    parser.add_argument('--batch_size', type=int, default=TIMITConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=TIMITConfig.epochs)
    parser.add_argument('--alpha', type=float, default=TIMITConfig.alpha)
    parser.add_argument('--beta', type=float, default=TIMITConfig.beta)
    parser.add_argument('--gamma', type=float, default=TIMITConfig.gamma)
    parser.add_argument('--hidden_size', type=float, default=TIMITConfig.hidden_size)
    parser.add_argument('--gpu', type=int, default=TIMITConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=TIMITConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=TIMITConfig.model_checkpoint)
    parser.add_argument('--noise_dataset_path', type=str, default=TIMITConfig.noise_dataset_path)

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Testing Model on NISP Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # hyperparameters and details about the model 
    HPARAMS = {
        'data_path' : hparams.data_path,
        'speaker_csv_path' : hparams.speaker_csv_path,
        'data_wav_len' : hparams.timit_wav_len,
        'data_batch_size' : hparams.batch_size,
        'data_wav_augmentation' : 'Random Crop, Additive Noise',
        'data_label_scale' : 'Standardization',

        'training_optimizer' : 'Adam',
        'training_lr' : TIMITConfig.lr,
        'training_lr_scheduler' : '-',

        'model_hidden_size' : hparams.hidden_size,
        'model_alpha' : hparams.alpha,
        'model_beta' : hparams.beta,
        'model_gamma' : hparams.gamma,
        'model_architecture' : 'wav2vec + soft-attention',
    }

    # Testing Dataset
    test_set = TIMITDataset(
        wav_folder = os.path.join(HPARAMS['data_path'], 'TEST'),
        csv_file = HPARAMS['speaker_csv_path'],
        wav_len = HPARAMS['data_wav_len'],
        is_train=False
    )

    csv_path = HPARAMS['speaker_csv_path']
    df = pd.read_csv(csv_path)
    h_mean = df['height'].mean()
    h_std = df['height'].std()
    a_mean = df['age'].mean()
    a_std = df['age'].std()

    #Testing the Model
    if hparams.model_checkpoint:
        model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=HPARAMS)
        model.eval()
        height_pred = []
        height_true = []
        age_pred = []
        age_true = []
        gender_pred = []
        gender_true = []


        # i = 0 
        for batch in tqdm(test_set):
            x, y_h, y_a, y_g = batch
            y_hat_h, y_hat_a, y_hat_g = model(x)

            height_pred.append((y_hat_h*h_std+h_mean).item())
            age_pred.append((y_hat_a*a_std+a_mean).item())
            gender_pred.append(y_hat_g>0.5)

            height_true.append((y_h*h_std+h_mean).item())
            age_true.append(( y_a*a_std+a_mean).item())
            gender_true.append(y_g)

            # if i> 5: break
            # i += 1
        female_idx = np.where(np.array(gender_true) == 1)[0].reshape(-1).tolist()
        male_idx = np.where(np.array(gender_true) == 0)[0].reshape(-1).tolist()

        height_true = np.array(height_true)
        height_pred = np.array(height_pred)
        age_true = np.array(age_true)
        age_pred = np.array(age_pred)


        hmae = mean_absolute_error(height_true[male_idx], height_pred[male_idx])
        hrmse = mean_squared_error(height_true[male_idx], height_pred[male_idx], squared=False)
        amae = mean_absolute_error(age_true[male_idx], age_pred[male_idx])
        armse = mean_squared_error(age_true[male_idx], age_pred[male_idx], squared=False)
        print(hrmse, hmae, armse, amae)

        hmae = mean_absolute_error(height_true[female_idx], height_pred[female_idx])
        hrmse = mean_squared_error(height_true[female_idx], height_pred[female_idx], squared=False)
        amae = mean_absolute_error(age_true[female_idx], age_pred[female_idx])
        armse = mean_squared_error(age_true[female_idx], age_pred[female_idx], squared=False)
        print(hrmse, hmae, armse, amae)

        print(accuracy_score(gender_true, gender_pred))


