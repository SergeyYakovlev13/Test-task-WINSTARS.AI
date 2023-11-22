# Test-task-WINSTARS.AI 

'data_analysys.ipynb' - jupyter notebook with exploratory data analysis of the dataset;

'model.py' - file with model architecture;

'metrics.py' - file with functions for necessary metrics (Dice and IoU);

'variables.py' - file with defined variables, which will be used in several other packages; 

'train.py' - file for training model;

'test.py' - file for testing pretrained model.

## To run task
  1. Save this repo.
  2. Install all requirements.
### To train model
  1. Download necessary train_ship_segmentations_v2.csv file from https://drive.google.com/drive/folders/1MWIJo6DEsK8m5IU9mZGs47BTwN6nCCz9?hl=ru.
  2. Open file 'train.py'.
  3. Change all the paths on those, which corresponds to your situation.
  4. Run the file.
### To test model
  1. Download pretrained model model.h5 from https://drive.google.com/drive/folders/1PEwqc017MtS6P-mibECO5ChlUSg7zQjt?hl=ru.
  2. Open file 'test.py'.
  3. Change all the paths on those, which corresponds to your situation.
  4. Run code.
