class TIMITConfig(object):
    # path to the unzuipped TIMIT data folder
    data_path = '/home/n1900235d/SpeakerProfiling/TimitDataset/wav_data'

    # path to csv file containing age, heights of timit speakers
    speaker_csv_path = 'src/Dataset/data_info_height_age.csv'

    # length of wav files for training and testing
    timit_wav_len = 16000 * 5

    batch_size = 128
    epochs = 100
    
    # loss = alpha * height_loss + beta * age_loss + gamma * gender_loss
    alpha = 1
    beta = 1
    gamma = 1

    # hidden dimension of LSTM and Dense Layers
    hidden_size = 128

    # No of GPUs for training and no of workers for datalaoders
    gpu = '-1'
    n_workers = 4

    # model checkpoint to continue from
    model_checkpoint = None

    # noise dataset for augmentation
    noise_dataset_path = '/home/n1900235d/INTERSPEECH/NoiseDataset'

    # LR of optimizer
    lr = 1e-3


class NISPConfig(object):
    # path to the unzuipped TIMIT data folder
    data_path = '/home/n1900235d/SpeakerProfiling/TimitDataset/wav_data'

    # path to csv file containing age, heights of timit speakers
    speaker_csv_path = '/home/shangeth/NISP/dataset/NISP-Dataset-master/total_spkrinfo.list'

    # length of wav files for training and testing
    timit_wav_len = 16000 * 5

    batch_size = 128
    epochs = 100
    
    # loss = alpha * height_loss + beta * age_loss + gamma * gender_loss
    alpha = 1
    beta = 1
    gamma = 1

    # hidden dimension of LSTM and Dense Layers
    hidden_size = 128

    # No of GPUs for training and no of workers for datalaoders
    gpu = '-1'
    n_workers = 4

    # model checkpoint to continue from
    model_checkpoint = None

    # noise dataset for augmentation
    noise_dataset_path = '/home/n1900235d/INTERSPEECH/NoiseDataset'

    # LR of optimizer
    lr = 1e-3