from torch.utils.data import Dataset
import os
import pandas as pd
import torch
import numpy as np

import torchaudio
import wavencoder
import random

class NISPDataset(Dataset):
    def __init__(self,
        wav_folder,
        csv_file,
        wav_len=48000,
        is_train=True,
        noise_dataset_path=None
        ):

        self.wav_folder = wav_folder
        self.files = os.listdir(self.wav_folder)
        self.csv_file = csv_file
        self.df = pd.read_csv(self.csv_file, sep=' ')
        self.is_train = is_train
        self.wav_len = wav_len
        self.noise_dataset_path = noise_dataset_path

        self.gender_dict = {'Male': 0,
                            'Female': 1}

        self.train_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len, pad_position='random', crop_position='random')
            ])

        # Pad/Crop from the center
        self.test_transform = wavencoder.transforms.Compose([
            wavencoder.transforms.PadCrop(pad_crop_length=self.wav_len)
            ])

        if self.noise_dataset_path:
            self.noise_transform = wavencoder.transforms.AdditiveNoise(self.noise_dataset_path)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file = self.files[idx]
        id = file[:8]

        wav, _ = torchaudio.load(os.path.join(self.wav_folder, file))

        if self.is_train:
            wav = self.train_transform(wav)

            # apply noise to wav 50% of time
            if self.noise_dataset_path and random.random() < 0.5:
                wav = self.noise_transform(wav)
        else:
            wav = self.test_transform(wav)
        
        df_row = self.df[self.df['Speaker_ID'] == id]
        gender = self.gender_dict[df_row['Gender'].item()]
        height = df_row['Height'].item()
        shoulder_size = df_row['Shoulder_size'].item()
        waist_size = df_row['Waist_size'].item()
        weight = df_row['Weight'].item()
        age = df_row['Age'].item()

        height = (height - self.df['Height'].mean())/self.df['Height'].std()
        age = (age - self.df['Age'].mean())/self.df['Age'].std()

        if type(wav).__module__ == np.__name__:
            wav = torch.tensor(wav)
        return wav, height, age, gender
