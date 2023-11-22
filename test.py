import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from variables import img_size,n_channels
from metrics import Dice,IoU

#model_path variable is path to directory, where model is saved(may be necessary to change).
model_path='model.h5'
model=tf.keras.models.load_model(model_path,custom_objects={"Dice":Dice,"IoU":IoU})

#numb1 is number of images, which will be saved to array, and from which will be chosen images for inference.
numb1=1000
#numb2 is number of images, chosen for inference.
numb2=6

#Creating array for images, and saving them into array.
if n_channels==3:
    images=np.zeros(shape=(numb1,img_size,img_size,n_channels),dtype='uint8')
if n_channels==1:
    images=np.zeros(shape=(numb1,img_size,img_size),dtype='uint8')
#images_path corresponds to path to directory with images(may be necessary to change).
images_path='test_images/'
i=0
for image in os.listdir(images_path):
    if n_channels==3:
        img=cv2.imread(images_path+image)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    if n_channels==1:
        img=cv2.imread(images_path+image,0)
    img=cv2.resize(img,(img_size,img_size))
    images[i]=img
    i+=1
    if i>=numb1:
        break

#Creating predicted masks(you can change threshold, if you want).
threshold=0.5
masks_pred=1*(model.predict(images)>threshold)
masks_pred[masks_pred==1]=255

#Generating index of images for inference(may repeat, generally), and showing examples of masks, predicted for images by model.
ind=np.random.randint(0,numb1,numb2)
fig=plt.figure(figsize=(numb2//2,16))
for i in range(numb2):
    ax1=fig.add_subplot(numb2,2,2*i+1)
    ax1.axis('off')
    ax1.set_title('Image',fontsize=7)
    if n_channels==3:
        ax1.imshow(images[ind[i]])
    if n_channels==1:
        ax1.imshow(images[ind[i]],cmap='gray')
    ax2=fig.add_subplot(numb2,2,2*i+2)
    ax2.axis('off')
    ax2.set_title('Predicted mask',fontsize=7)
    ax2.imshow(masks_pred[ind[i]],cmap='gray')
plt.show()
