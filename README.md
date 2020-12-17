# Overview
This project is the work of:
 * Dimitris Halatsis
 * Orest Gkini
 * Theofilos Belmpas

It's contains code that tries to solve a 'Road Segmentation' challenge in [ai-crowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation/leaderboards)

## Code structure
All the code is located under the `script` folder.

* [Models folder](scripts/models): Contains all the models we have tried (FCN, UNet, ResUNet)
* [Data Utils folder](scripts/utils): Contains code used for loading, storing and processing images
* [General Utils folder](scripts/utils): Contains util code like for the images, or the models (saving & loading)
* [run.py](scripts/run.py): Contains the code that was used to create our best submission (id=109641)

@NOTE: Data are included to this project, but they must be unzipped manually and then alter some paths in [run.py](scripts/run.py)

## Preliminaries

### Unzipping the data
You must unzip the data found in [training.zip](data/training.zip) file in whatever path you like (decompressed size: 34.2 MB).
Also, unzip the test data found in [test_set_images.zip](data/test_set_images.zip) file if you would like to generate predictions for the test set. (decompressed size: 39.5 MB).

### Change paths in [run.py](scripts/run.py)
Then, alter the paths located in lines 36 and 42 of [run.py](scripts/run.py) to match the paths of the unzipped train & test images respectively.


## Run code

Before running the code make sure that all the packages mentioned in `requirements.txt` exist. Also make sure that all data are present and paths are set correctly (see [preliminaries](#preliminaries))

@NOTE: Please think about using a GPU accelerated environment, otherwise it make take some hours to run.

### Running the [run.py](scripts/run.py)
Use the following command form the root of the folder
```bash
python run.py -m scripts
```

## Load a pre-trained model
Find our pre-trained model in our [google drive](https://drive.google.com/file/d/1-3mmSxURo_iYPbh3a7uVKb9vkKqYbQT6/view?usp=sharing) and download it locally (size= 237 MB)

You can use this code to load our pre-trained model
```python
from scripts.utils.model import load
means = [0.33526483, 0.33185714, 0.29748788]
stds = [0.19387639, 0.18752979, 0.18748713]

# Define a model
unet = UNet(3,2)

# Load our best model
load(unet, path='the_path/you_downloaded/the_model')

# Create transformation for the images
loaded_test_trans = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(means, stds)
])

# ... Load some img

# Predict with our model
unet.pred(img, loaded_test_trans)
```
@NOTE: make sure to place this code outside `scripts` folder so you can import all modules using `from scripts.<something> import ...`