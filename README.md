# Speaker Profiling

This Repository contains the code for estimating the Age, Height and Gender of a speaker with their speech signal. The repository experiments with both TIMIT and NISP Dataset.

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

# NISP Dataset
git clone https://github.com/iiscleap/NISP-Dataset.git
```

### Prepare the dataset for training and testing
```
# TIMIT Dataset
python prepare_timit_data.py --path='path to timit data folder'

# NISP Dataset
python prepare_nisp_data.pt --nisp_repo_path='path to nisp data repo folder'
```

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


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)