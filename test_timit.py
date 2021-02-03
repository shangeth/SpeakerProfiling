from argparse import ArgumentParser
from multiprocessing import Pool
import os

from TIMIT.dataset import TIMITDataset
from TIMIT.model import Wav2VecModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch
import torch.utils.data as data


if __name__ == "__main__":

    parser = ArgumentParser(add_help=True)
    parser.add_argument('--data_path', type=str, default='/home/shangeth/speaker_profiling/datadir/wav_data')
    parser.add_argument('--speaker_csv_path', type=str, default='/home/shangeth/speaker_profiling/src/data_info_height_age.csv')
    parser.add_argument('--timit_wav_len', type=int, default=16000*5)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--hidden_size', type=float, default=128)
    parser.add_argument('--gpu', type=int, default="1")
    parser.add_argument('--n_workers', type=int, default=int(int(Pool()._processes)*0.1))
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--noise_dataset_path', type=str, default='/home/shangeth/speaker_profiling/noise_datadir/noises')

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
        'training_lr' : 1e-3,
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
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=False, 
        num_workers=hparams.n_workers
    )


    #Testing the Model
    if hparams.model_checkpoint:
        model = Wav2VecModel.load_from_checkpoint(hparams.model_checkpoint, HPARAMS=HPARAMS)
        trainer = pl.Trainer(fast_dev_run=hparams.dev, 
                            gpus=hparams.gpu, 
                            )

        print('\nTesting on TIMIT Dataset:\n')
        trainer.test(model, test_dataloaders=testloader)
    else:
        print('Model check point for testing is not provided!!!')