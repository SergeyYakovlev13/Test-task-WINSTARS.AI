# Test-task-WINSTARS.AI
## Description
This basic purpose of this project is to make a model that will make a segmentation of an input image. To accomplish this U-Net model was trained on a kaggle competitions Airbus Ship Detection Challenge Dataset. 

'data_analysys.jpynb' - jupyter notebook with exploratory data analysis of the dataset;

'model.py' - file with model architecture;

'metrics.py' - file with functions for necessary metrics (Dice and IoU);

'variables.py' - file with defined variables, which will be used in several other packages; 

'train.py' - file for training model;

'test.py' - file for testing pretrained model.

## To run task
  1. Save this repo.
  2. Install all requirements.
### To train model
  1. Open file 'train.py'.
  2. In the fifth row replace path to your dataset path (format of paths have to be path/to/images/* and path/to/masks/*).
  3. Run the file.
### To test model
  1. Download pretrained model model.h5 from https://drive.google.com/drive/folders/1PEwqc017MtS6P-mibECO5ChlUSg7zQjt?hl=ru.
  2. Open file 'test.py'.
  3. Replase paths in rows 9, 14, 15, 18.
  4. Run code.
