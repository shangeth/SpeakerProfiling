# Speaker Profiling

This Repository contains the code for estimating the Age, Height and Gender of a speaker with their speech signal. The repository experiments with both TIMIT and NISP Dataset.

![model architeture](assets/wav2vecframework.PNG)

[DEMO Colab Notebook](https://colab.research.google.com/drive/1WDBtlhg87BiPlg-IrIiFxyY5eaOVqkob?usp=sharing)

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required packages for preparing the dataset, training and testing the model.

```bash
pip install -r requirements.txt
```

## Usage

### Download the dataset
```
# Timit Dataset
wget https://data.deepai.org/timit.zip
unzip timit.zip -d 'path to timit data folder'

# NISP Dataset
git clone https://github.com/iiscleap/NISP-Dataset.git
```

### Prepare the dataset for training and testing
```
# TIMIT Dataset
python TIMIT/prepare_timit_data.py --path='path to timit data folder'

# NISP Dataset
python NISP/prepare_nisp_data.pt --nisp_repo_path='path to nisp data repo folder'
```

### Update Config and Logger
Update the config.py file to update the batch_size, gpus, lr, etc and change the preferred logger in train_.py files

### Training(Dev Model, to make sure everything is set as expected for training) 
```
# TIMIT Dataset
python train_timit.py --dev=True --data_path='path to final data folder'

# NISP Dataset
python train_nisp.py --dev=True --data_path='path to final data folder'
```

### Training(also check for other arguments in the train_....py file)
```
# TIMIT Dataset
python train_timit.py --data_path='path to final data folder'

# NISP Dataset
python train_nisp.py --data_path='path to final data folder'
```

### Test the Model
```
# TIMIT Dataset
python test_timit.py --data_path='path to final data folder' --model_checkpoint='path to saved model checkpoint'

# NISP Dataset
python test_nisp.py --data_path='path to final data folder' --model_checkpoint='path to saved model checkpoint'
```

## Results

[Wandb Runs](https://wandb.ai/shangeth/SpeakerProfiling?workspace=user-shangeth)
### TIMIT Baseline
| Model                                  	| Height RMSE 	|        	| Height MAE 	|        	| Age RMSE 	|        	| Age MAE 	|        	| Gender Acc 	|
|----------------------------------------	|-------------	|--------	|------------	|--------	|----------	|--------	|---------	|--------	|------------	|
|                                        	| Male        	| Female 	| Male       	| Female 	| Male     	| Female 	| Male    	| Female 	|            	|
| MFCC_LSTM-Attn                         	| 7.5         	| 6.6    	| 5.5        	| 5.2    	| 7.7      	| 8.4    	| 5.6     	| 5.9    	| 0.975      	|
| MelSpec_LSTM-Attn                      	| 7.7         	| 8.1    	| 5.8        	| 6.5    	| 7.7      	| 8.7    	| 5.5     	| 6.1    	| 0.669      	|
| MFCC_CNN-LSTM-Attn                     	| 7.5         	| 6.8    	| 5.7        	| 5.3    	| 8.2      	| 8.7    	| 5.4     	| 6.1    	| 0.989      	|
| MelSpec_CNN-LSTM-Attn                  	| 7.5         	| 7.4    	| 5.8        	| 5.8    	| 8.2      	| 8.4    	| 5.8     	| 5.9    	| 0.96       	|
| wav2vec(no-finetune)-LSTM-Attn         	| 7.4         	| 6.4    	| 5.5        	| 5.1    	| 7.2      	| 8.2    	| 5.0     	| 5.7    	| 0.994      	|
| wav2vec(finetune 56)-LSTM-Attn         	| 7.5         	| 6.2    	| 5.5        	| 4.9    	| 7.5      	| 7.9    	| 5.5     	| 5.7    	| 0.994      	|
| wav2vec(finetune 6)-LSTM-Attn          	| 7.6         	| 6.7    	| 5.6        	| 5.3    	| 7.0      	| 8.2    	| 4.9     	| 5.6    	| 0.993      	|
| wav2vec(finetune 56)-LSTM-Attn(Only H) 	| 7.4         	| 6.2    	| 5.6        	| 4.9    	|          	|        	|         	|        	|            	|
| multi-scale-cnn(Only H) 	| 7.5         	| 6.1    	| 5.9        	| 4.7    	|          	|        	|         	|        	|            	|

### TIMIT Previous Results
|        Model        	|  Height 	|  RMSE  	|  Height 	|   MAE  	|  Age 	|  RMSE  	|  Age 	|   MAE  	| Gender Acc 	|
|:-------------------:	|:-------:	|:------:	|:-------:	|:------:	|:----:	|:------:	|:----:	|:------:	|:----------:	|
|                     	|   Male  	| Female 	|   Male  	| Female 	| Male 	| Female 	| Male 	| Female 	|            	|
|       [1] 2019      	|   6.85  	|  6.29  	|    -    	|    -   	|  7.6 	|  8.63  	|   -  	|    -   	|            	|
|  [2] 2016 (fusion)  	|   **6.7**   	|   6.1  	|  **5.0**   	|   5.0  	|  7.8 	|   8.9  	|  5.5 	|   6.5  	|            	|
| [2] 2016 (baseline) 	|   7.0   	|   6.5  	|   5.3   	|   5.2  	|  8.1 	|   9.1  	|  5.7 	|   6.2  	|            	|
|       [3] 2020      	|    -    	|    -   	|    -    	|    -   	| 7.24 	|  8.12  	| 5.12 	|  **5.29**  	|    **0.996**   	|
|       [4] 2009      	|   6.8   	|   6.3  	|   5.3   	|   5.1  	|   -  	|    -   	|   -  	|    -   	|            	|


### NISP
|    Model   	| Height 	|  RMSE  	| Height 	|   MAE  	|  Age 	|  RMSE  	|  Age 	|   MAE  	| Gender Acc 	|
|:----------:	|:------:	|:------:	|:------:	|:------:	|:----:	|:------:	|:----:	|:------:	|:----------:	|
|            	|  Male  	| Female 	|  Male  	| Female 	| Male 	| Female 	| Male 	| Female 	|            	|
|   [5] TMP  	| 6.17   	| 6.93   	| 5.22   	| 5.30   	| 5.60 	| 5.57   	| 4.40 	| 4.42   	|            	|
| [5] Comb-3 	| 6.13   	| 6.70   	| 5.16   	| 5.30   	| 5.63 	| 4.99   	| 3.80 	| 3.76   	|            	|
| Our Method 	| 6.49   	| 6.37   	| 5.32   	| 5.12   	| 5.48 	| 5.71   	| 3.70 	| 4.22   	| 0.984      	|

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)

## Reference
- [1] S. B. Kalluri, D. Vijayasenan and S. Ganapathy, "A Deep Neural Network Based End to End Model for Joint Height and Age Estimation from Short Duration Speech," ICASSP 2019 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Brighton, United Kingdom, 2019, pp. 6580-6584, doi: 10.1109/ICASSP.2019.8683397.
- [ 2 ]  Rita Singh, Bhiksha Raj, and James Baker, “Short-term analysis for estimating physical parameters of speakers,” in Proc. of IWBF. IEEE, 2016, pp. 1–6
- [ 3 ] Joint gender and age estimation based on speech signals using x-vectors and transfer learning ICASSP 2021.
- [ 4 ] Mporas, I., Ganchev, T. Estimation of unknown speaker’s height from speech. Int J Speech Technol 12, 149–160 (2009). https://doi.org/10.1007/s10772-010-9064-2

