from config import TIMITConfig
from argparse import ArgumentParser
from multiprocessing import Pool
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning import Trainer


import torch
import torch.utils.data as data
# torch.use_deterministic_algorithms(True)


# SEED
SEED=100
pl.utilities.seed.seed_everything(SEED)
torch.manual_seed(SEED)


from TIMIT.dataset import TIMITDataset
if TIMITConfig.training_type == 'H':
    from TIMIT.lightning_model_h import LightningModel
else:
    from TIMIT.lightning_model import LightningModel

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
    parser.add_argument('--lr', type=float, default=TIMITConfig.lr)
    parser.add_argument('--gpu', type=int, default=TIMITConfig.gpu)
    parser.add_argument('--n_workers', type=int, default=TIMITConfig.n_workers)
    parser.add_argument('--dev', type=str, default=False)
    parser.add_argument('--model_checkpoint', type=str, default=TIMITConfig.model_checkpoint)
    parser.add_argument('--noise_dataset_path', type=str, default=TIMITConfig.noise_dataset_path)
    parser.add_argument('--model_type', type=str, default=TIMITConfig.model_type)
    parser.add_argument('--training_type', type=str, default=TIMITConfig.training_type)
    parser.add_argument('--data_type', type=str, default=TIMITConfig.data_type)

    parser = pl.Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()
    print(f'Training Model on TIMIT Dataset\n#Cores = {hparams.n_workers}\t#GPU = {hparams.gpu}')

    # Training, Validation and Testing Dataset
    ## Training Dataset
    train_set = TIMITDataset(
        wav_folder = os.path.join(hparams.data_path, 'TRAIN'),
        hparams = hparams
    )
    ## Training DataLoader
    trainloader = data.DataLoader(
        train_set, 
        batch_size=hparams.batch_size, 
        shuffle=True, 
        num_workers=hparams.n_workers
    )
    ## Validation Dataset
    valid_set = TIMITDataset(
        wav_folder = os.path.join(hparams.data_path, 'VAL'),
        hparams = hparams,
        is_train=False
    )
    ## Validation Dataloader
    valloader = data.DataLoader(
        valid_set, 
        batch_size=1,
        # hparams.batch_size, 
        shuffle=False, 
        num_workers=hparams.n_workers
    )
    ## Testing Dataset
    test_set = TIMITDataset(
        wav_folder = os.path.join(hparams.data_path, 'TEST'),
        hparams = hparams,
        is_train=False
    )
    ## Testing Dataloader
    testloader = data.DataLoader(
        test_set, 
        batch_size=1,
        # hparams.batch_size, 
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

    model = LightningModel(vars(hparams))

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
                patience=20,
                verbose=True,
                mode='min'
                )
        ],
        logger=logger,
        resume_from_checkpoint=hparams.model_checkpoint,
        distributed_backend='ddp'
        )

    trainer.fit(model, train_dataloader=trainloader, val_dataloaders=valloader)

    # print('\n\nCompleted Training...\nTesting the model with checkpoint -', checkpoint_callback.best_model_path)
    # model = LightningModel.load_from_checkpoint(checkpoint_callback.best_model_path)
    # trainer.test(model, test_dataloaders=testloader)
