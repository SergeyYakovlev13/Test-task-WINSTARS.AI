# Test-task-WINSTARS.AI 
## Description
 Main goal of this project - to build semantic segmentation model, trained on a dataset from Kaggle Airbus Ship Detection Challenge competition. 

 As a result, was implemented U-Net model, which recieved on a validation set Dice of 85,64% and IoU of 74,96%, which are quite satisfiable results. 

'data_analysys.ipynb' - jupyter notebook with exploratory data analysis of the dataset;

'model.py' - file with model architecture;

'metrics.py' - file with functions for necessary metrics (Dice and IoU);

'variables.py' - file with defined variables, which will be used in several other packages; 

'train.py' - file for training model;

'test.py' - file for testing pretrained model;

'alternative.ipynb' - jupyter notebook, which was created from other .py files in this project, and runned on Kaggle platform (as lon as it provides free GPU-usage quota, for faster and more qualified learning), which might be for some people more comfortable way of analysing this code.  

## To run task
  1. Save this repository.
  2. Install all requirements.
### To train model
  1. Download necessary train_ship_segmentations_v2.csv file from https://drive.google.com/drive/folders/1MWIJo6DEsK8m5IU9mZGs47BTwN6nCCz9?hl=ru.
  2. Open file 'train.py'.
  3. Change all the paths on those, which corresponds to your situation(also download images for training, if you still haven't done it).
  4. If you don't have enough computational resources, you can take significantly less images into dataset, and try learning for a less epochs, by changing corresponding variables.
  5. Run the file.
### To test model
  1. Download pretrained model model.h5 from https://drive.google.com/drive/folders/1PEwqc017MtS6P-mibECO5ChlUSg7zQjt?hl=ru.
  2. Open file 'test.py'.
  3. Change all the paths on those, which corresponds to your situation(also download images for testing, if you still haven't done it).
  4. Run code.
