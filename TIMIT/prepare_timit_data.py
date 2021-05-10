import os
import shutil
import argparse
from sklearn.model_selection import train_test_split

my_parser = argparse.ArgumentParser(description='Path to the TIMIT dataset folder')
my_parser.add_argument('--path',
                       metavar='path',
                       type=str,
                       help='the path to dataset folder')
args = my_parser.parse_args()


original_data_dir = args.path
original_wav_path = os.path.join(original_data_dir, 'data')

final_data_path = os.path.join(original_data_dir, 'wav_data')
final_test_path = os.path.join(final_data_path, 'TEST')
final_train_path = os.path.join(final_data_path, 'TRAIN')
final_val_path = os.path.join(final_data_path, 'VAL')
final_phn_path = os.path.join(final_data_path, 'phn')
final_txt_path = os.path.join(final_data_path, 'txt')

if not os.path.exists(final_data_path):
    os.mkdir(final_data_path)
    os.mkdir(final_train_path)
    os.mkdir(final_val_path)
    os.mkdir(final_test_path)
    os.mkdir(final_phn_path)
    os.mkdir(final_txt_path)

    for type in os.listdir(original_wav_path):
        for fol in os.listdir(os.path.join(original_wav_path, type)):
            for id in os.listdir(os.path.join(original_wav_path, type, fol)):
                for i, file in enumerate(os.listdir(os.path.join(original_wav_path, type, fol, id))):
                    if file.endswith('.WAV'):
                        src = os.path.join(original_wav_path, type, fol, id, file)
                        dst = os.path.join(final_data_path, type, f'{id}_{file}')
                        shutil.copy(src, dst)
                    elif file.endswith('.PHN'):
                        src = os.path.join(original_wav_path, type, fol, id, file)
                        dst = os.path.join(final_phn_path, f'{id}_{file}')
                    elif file.endswith('.TXT'):
                        src = os.path.join(original_wav_path, type, fol, id, file)
                        dst = os.path.join(final_txt_path, f'{id}_{file}')


M_speakers_id = []
F_speakers_id = []

for file in os.listdir(final_train_path):
    id = file.split('_')[0]
    if id.startswith('M'):
        M_speakers_id.append(id)
    elif id.startswith('F'):
        F_speakers_id.append(id)

M_speakers_id = list(set(M_speakers_id))
F_speakers_id = list(set(F_speakers_id))


train_m, val_m = train_test_split(M_speakers_id, test_size=0.15, random_state = 1)
train_f, val_f = train_test_split(F_speakers_id, test_size=0.15, random_state = 2)


train_ids = train_m + train_f
val_ids = val_m + val_f


for file in os.listdir(final_train_path):
    src = os.path.join(final_train_path, file)
    id =  file.split('_')[0]
    if id in val_ids:
        dst = os.path.join(final_val_path, file)
        shutil.move(src, dst)

print('Wav Data saved at ', final_data_path)



