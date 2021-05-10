from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder


class TIMITDataset(Dataset):
    def __init__(self,
    wav_folder,
    hparams,
    is_train=True,
    ):
        self.wav_folder = wav_folder
        self.files = os.listdir(self.wav_folder)
        self.csv_file = hparams.speaker_csv_path
        self.df = pd.read_csv(self.csv_file)
        self.is_train = is_train
        self.wav_len = hparams.timit_wav_len
        self.noise_dataset_path = hparams.noise_dataset_path
        self.data_type = hparams.data_type

        self.speaker_list = self.df.loc[:, 'ID'].values.tolist()
        self.df.set_index('ID', inplace=True)
        self.gender_dict = {'M' : 0, 'F' : 1}

        if self.noise_dataset_path:
            self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='random', crop_position='random'),
                wavencoder.transforms.AdditiveNoise(self.noise_dataset_path, p=0.5),
                wavencoder.transforms.Clipping(p=0.5),
                ])
        else:
            self.train_transform = wavencoder.transforms.Compose([
                wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='left', crop_position='random'),
                wavencoder.transforms.Clipping(p=0.5),
                ])

        self.test_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='left', crop_position='center')
            ])

        if self.data_type == 'spectral':
            # self.spectral_transform = torchaudio.transforms.MelSpectrogram(normalized=True)
            self.spectral_transform = torchaudio.transforms.MFCC(n_mfcc=40, log_mels=True)
            self.spec_aug = wavencoder.transforms.Compose([  
                torchaudio.transforms.FrequencyMasking(5),
                torchaudio.transforms.TimeMasking(5),
            ])

    def __len__(self):
        return len(self.files)

    def get_age(self, idx):
        rec_date = self.df.loc[idx, 'RecDate'].split('/')
        birth_date = self.df.loc[idx, 'BirthDate'].split('/')
        m1, d1, y1 = [int(x) for x in birth_date]
        m2, d2, y2 = [int(x) for x in rec_date]
        return y2 - y1 - ((m2, d2) < (m1, d1))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file.split('_')[0][1:]
        g_id = file.split('_')[0]

        gender = self.gender_dict[self.df.loc[id, 'Sex']]
        height = self.df.loc[id, 'height']
        age =  self.df.loc[id, 'age']
        # self.get_age(id)

        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file))
        if self.is_train:
            wav = self.train_transform(wav)  
            if self.data_type == 'spectral':
                wav = self.spectral_transform(wav)
                wav = self.spec_aug(wav)

        else:
            # wav = self.test_transform(wav)
            if self.data_type == 'spectral':
                wav = self.spectral_transform(wav)
        
        height = (height - self.df['height'].mean())/self.df['height'].std()
        age = (age - self.df['age'].mean())/self.df['age'].std()
<<<<<<< HEAD
        return wav, height, age, gender
=======

        if type(wav).__module__ == np.__name__:
            wav = torch.tensor(wav)
        
        # wav = self.spec_aug(self.spectral_transform(wav))/100
        # print(wav.min(), wav.max(), wav.mean(), wav.std())
        return wav, height, age, gender
>>>>>>> 9533cd02cfa4ddb9aee40945d53e3354b5d5d960
