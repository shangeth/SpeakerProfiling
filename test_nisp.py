from argparse import ArgumentParser
from multiprocessing import Pool
import os

from NISP.dataset import NISPDataset
from NISP.lightning_model import LightningModel

import pytorch_lightning as pl

from config import NISPConfig

import torch
import torch.utils.data as data


if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default=NISPConfig.data_path)
    parser.add_argument('--speaker_csv_path', type=str, default=NISPConfig.speaker_csv_path)
    parser.add_argument('--timit_wav_len', type=int, default=NISPConfig.timit_wav_len)
    parser.add_argument('--batch_size', type=int, default=NISPConfig.batch_size)
    parser.add_argument('--epochs', type=int, default=NISPConfig.epochs)
    parser.add_argument('--alpha', type=float, default=NISPConfig.alpha)
    parser.add_argument('--beta', type=float, default=NISPConfig.beta)
    parser.add_argument('--gamma', type=float, default=NISPConfig.gamma)
    parser.add_argument('--hidden_size', type=float, default=NISPConfig.hidden_size)
    parser.add_argument('--gpu', type=int, default=NISPConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=NISPConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=NISPConfig.model_checkpoint)
    parser.add_argument('--noise_dataset_path', type=str, default=NISPConfig.noise_dataset_path)

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
        'training_lr' : NISPConfig.lr,
        'training_lr_scheduler' : '-',

        'model_hidden_size' : hparams.hidden_size,
        'model_alpha' : hparams.alpha,
        'model_beta' : hparams.beta,
        'model_gamma' : hparams.gamma,
        'model_architecture' : 'wav2vec + soft-attention',
    }

    # Testing Dataset
    test_set = NISPDataset(
        wav_folder = os.path.join(HPARAMS['data_path'], 'TEST'),
        csv_file = HPARAMS['speaker_csv_path'],
        wav_len = HPARAMS['data_wav_len'],
        is_train=False
    )
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=False, 
        num_workers=hparams.n_workers
    )


    #Testing the Model
    if hparams.model_checkpoint:
        model = LightningModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=HPARAMS)
        trainer = pl.Trainer(fast_dev_run=hparams.dev, 
                            gpus=hparams.gpu, 
                            )

        print('\nTesting on NISP Dataset:\n')
        trainer.test(model, test_dataloaders=testloader)
    else:
        print('Model check point for testing is not provided!!!')