import os
import shutil
import tarfile
import subprocess
import shlex
import pandas as pd
from sklearn.model_selection import train_test_split
import librosa
import soundfile as sf
from tqdm import tqdm

parser = ArgumentParser(add_help=True)
parser.add_argument('--nisp_repo_path', type=str, default='/home/shangeth/NISP/dataset/NISP-Dataset-master')
parser = pl.Trainer.add_argparse_args(parser)
hparams = parser.parse_args()

####################################
# 1. Extract tar files 

data_path = hparams.nisp_repo_path
lang_dirs = [x for x in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, x))]

final_tar_path = os.path.join(data_path, 'final_tars')
if not os.path.exists(final_tar_path): os.mkdir(final_tar_path)

tmp_tar_file = os.path.join(final_tar_path, 'tmp_tar_file.tar.gz')
tmp_tar_file2 = os.path.join(final_tar_path, 'tmp_tar_file.tar')


for lang_fol in lang_dirs:
    sub_lang_dirs = os.listdir(os.path.join(data_path, lang_fol))
    sub_lang_dirs.remove('scripts')
    
    for sub_lang_fol in sub_lang_dirs:
        tmp_path = os.path.join(data_path, lang_fol, sub_lang_fol)
        os.chdir(tmp_path)
        cmd1 = "cat *.tar.gz.* > {}".format(tmp_tar_file)
        out = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print(out)

        os.chdir(final_tar_path)
        cmd2 = "gzip -d {}".format(tmp_tar_file)
        out = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print(out)

        cmd3 = "tar -xvzf {}".format(tmp_tar_file2)
        out = subprocess.run(cmd3, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        print(out)

        os.remove(tmp_tar_file2)
        print(tmp_path)


#########
# 2. Arrange wav files into Train/Test/Validation sets

wav_dir_path = data_path + '/final_tars/RECS/'
data_files_dir = data_path

final_data_dir = os.path.join(data_files_dir, 'final_data')
final_data_dir_train = os.path.join(final_data_dir, 'TRAIN')
final_data_dir_test = os.path.join(final_data_dir, 'TEST')
final_data_dir_val = os.path.join(final_data_dir, 'VAL')

if not os.path.exists(final_data_dir): 
    os.mkdir(final_data_dir)
    os.mkdir(final_data_dir_train)
    os.mkdir(final_data_dir_test)
    os.mkdir(final_data_dir_val)


speakers_df = pd.read_csv(os.path.join(data_files_dir, 'total_spkrinfo.list'), sep=' ')
train_df = pd.read_csv(os.path.join(data_files_dir, 'train_spkrID'), header=None).applymap(lambda x:x[:8])
test_df = pd.read_csv(os.path.join(data_files_dir, 'test_spkrID'), header=None).applymap(lambda x:x[:8])

train_speakers_list = train_df[0].to_list()
train_height_list = train_df[3].to_list()
test_speakers_list = test_df[0].to_list()

train_speakers_list, val_speakers_list, _, _ = train_test_split(train_speakers_list, train_height_list, test_size=0.2)

for fol in os.listdir(wav_dir_path):
    if os.path.isdir(os.path.join(wav_dir_path, fol)):
        for wavfile in os.listdir(os.path.join(wav_dir_path, fol)):
            id = wavfile[:8]
            if id in train_speakers_list:
                src = os.path.join(wav_dir_path, fol, wavfile)
                dst = os.path.join(final_data_dir_train, wavfile)
                shutil.copy(src, dst)
            
            elif id in test_speakers_list:
                src = os.path.join(wav_dir_path, fol, wavfile)
                dst = os.path.join(final_data_dir_test, wavfile)
                shutil.copy(src, dst)

            elif id in val_speakers_list:
                src = os.path.join(wav_dir_path, fol, wavfile)
                dst = os.path.join(final_data_dir_val, wavfile)
                shutil.copy(src, dst)


print(len(os.listdir(final_data_dir_train)), len(os.listdir(final_data_dir_test)), len(os.listdir(final_data_dir_val)))

##################
# 3. Resample wav files to 16k

def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

SAMPLE_RATE = 16000
data_path = data_path

src = os.path.join(data_path, 'final_data')
dst =  os.path.join(data_path, 'final_data_16k')
if not os.path.exists(dst): os.mkdir(dst)

copytree(src, dst)

for fol in os.listdir(dst):
    for f in tqdm(os.listdir(os.path.join(dst, fol))):
        file = os.path.join(dst, fol, f)
        y, sr = librosa.load(file)
        data = librosa.resample(y, sr, SAMPLE_RATE)
        sf.write(file, data, SAMPLE_RATE)

print('Final Path of 16k NISP Dataset = ', dst)