from argparse import ArgumentParser
from multiprocessing import Pool
import os

from TIMIT.dataset import TIMITDataset
from TIMIT.lightning_model import LightningModel
# from TIMIT.lightning_model_h import LightningModel

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer

from config import TIMITConfig
import torch
import torch.utils.data as data
# torch.use_deterministic_algorithms(True)


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
    print(f'Training Model on TIMIT Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

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

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = TIMITDataset(
        wav_folder = os.path.join(HPARAMS['data_path'], 'TRAIN'),
        csv_file = HPARAMS['speaker_csv_path'],
        wav_len = HPARAMS['data_wav_len'],
        noise_dataset_path = hparams.noise_dataset_path
    )
    ## Training DataLoader
    trainloader = data.DataLoader(
        train_set, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=True, 
        num_workers=hparams.n_workers
    )
    ## Validation Dataset
    valid_set = TIMITDataset(
        wav_folder = os.path.join(HPARAMS['data_path'], 'VAL'),
        csv_file = HPARAMS['speaker_csv_path'],
        wav_len = HPARAMS['data_wav_len'],
        is_train=False
    )
    ## Validation Dataloader
    valloader = data.DataLoader(
        valid_set, 
        batch_size=HPARAMS['data_batch_size'], 
        shuffle=False, 
        num_workers=hparams.n_workers
    )
    ## Testing Dataset
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

    print('Dataset Split (Train, Validation, Test)=', len(train_set), len(valid_set), len(test_set))


    # Training the Model
    # logger = TensorBoardLogger('TIMIT_logs', name='')
    logger = WandbLogger(
        name=TIMITConfig.run_name,
        project='SpeakerProfiling'
    )

    model = LightningModel(HPARAMS)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss', 
        mode='min',
        verbose=1)

    trainer = Trainer(
        fast_dev_run=hparams.dev, 
        gpus=hparams.gpu, 
        max_epochs=hparams.epochs, 
        checkpoint_callback=checkpoint_callback,
        callbacks=[
            EarlyStopping(
                monitor='val/loss',
                min_delta=0.00,
                patience=50,
                verbose=True,
                mode='min'
                )
        ],
        logger=logger,
        resume_from_checkpoint=hparams.model_checkpoint,
        distributed_backend='ddp'
        )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

    print('\n\nCompleted Training...\nTesting the model with checkpoint -', checkpoint_callback.best_model_path)
    model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    trainer.test(model, test_dataloaders=testloader)