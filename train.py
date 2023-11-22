import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from variables import img_size,n_channels
from model import segmentation_model
from metrics import IoU,Dice

#Saving DataFrame with information about datatset(may be necessary to change a path).
df_path='train_ship_segmentations_v2.csv'
df=pd.read_csv(df_path)

#Transforming DataFrame with information about datatset into suitable format for further work.
df['EncodedPixels'].fillna('0',inplace=True)
df=df.groupby(['ImageId'], as_index=False).agg({'EncodedPixels': ' '.join})

#Creating arrays for images and masks, and filling them.
#n_images corresponds to number of images with ships, and no_ships corresponds to number of images without ships in out dataset. 
n_images=25000
no_ships=3000
#May be nesessary to change path to directory with images.
path='images/' 
i=0
if n_channels==3:
    images=np.zeros(shape=(n_images+no_ships,img_size,img_size,n_channels),dtype='uint8')
if n_channels==1:
    images=np.zeros(shape=(n_images+no_ships,img_size,img_size),dtype='uint8')
masks=np.zeros(shape=(n_images+no_ships,img_size,img_size),dtype='uint8')
for image in df[df['EncodedPixels']!='0']['ImageId'].values:
    if n_channels==3:
        img=cv2.imread(path+image)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    elif n_channels==1:
        img=cv2.imread(path+image,0)
    h=img.shape[0]
    w=img.shape[1]
    mask=np.zeros(shape=(h,w))
    pix=df[df['ImageId']==image]['EncodedPixels'].values[0]
    pix=pix.split(' ')
    pix=[int(x) for x in pix]
    for j in range(0,len(pix),2):
        start=pix[j]
        move=pix[j+1]
        start_h=start//h
        start_v=start-1-start_h*h
        for k in range(move):
            if start_v+k<h:
                mask[start_v+k,start_h]=255
            else:
                mask[start_v+k-h,start_h+1]=255
    img=cv2.resize(img,(img_size,img_size))
    images[i]=img
    masks[i]=cv2.resize(mask,(img_size,img_size))
    i+=1
    if i>=n_images:
        break
for image in df[df['EncodedPixels']=='0']['ImageId'].values:
    if i%10==0:
        print(i)
    if n_channels==3:
        img=cv2.imread(path+image)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    elif n_channels==1:
        img=cv2.imread(path+image,0)
    h=img.shape[0]
    w=img.shape[1]
    mask=np.zeros(shape=(h,w))
    img=cv2.resize(img,(img_size,img_size))
    images[i]=img
    masks[i]=cv2.resize(mask,(img_size,img_size))
    i+=1
    if i>=n_images+no_ships:
        break

#Splitting dataset on train,test and validation, and making correct labels for masks
masks[masks>0]=1 
images_train,images_test,masks_train,masks_test=train_test_split(images,masks,train_size=0.8,random_state=1)
images_val,images_test,masks_val,masks_test=train_test_split(images_test,masks_test,test_size=0.5,random_state=1)

#Applying data augmentation(horizontal and vertical flip) + shuffling data after augmentation.
images_train=np.append(images_train,np.array([cv2.flip(image,1) for image in images_train]),axis=0)
masks_train=np.append(masks_train,np.array([cv2.flip(mask,1) for mask in masks_train]),axis=0)
images_train=np.append(images_train,np.array([cv2.flip(image,0) for image in images_train]),axis=0)
masks_train=np.append(masks_train,np.array([cv2.flip(mask,0) for mask in masks_train]),axis=0)
images_train,masks_train=shuffle(images_train,masks_train,random_state=1)

#Converting data to the format suitable for using in TensorFlow
images_train=tf.convert_to_tensor(images_train)
masks_train=tf.convert_to_tensor(masks_train)
images_val=tf.convert_to_tensor(images_val)
masks_val=tf.convert_to_tensor(masks_val)
images_test=tf.convert_to_tensor(images_test)
masks_test=tf.convert_to_tensor(masks_test)

#It is possible to change batch size, if it is necessary/you have a wish.
BATCH_SIZE=128
train_data=tf.data.Dataset.from_tensor_slices((images_train,masks_train)).batch(BATCH_SIZE)
val_data=tf.data.Dataset.from_tensor_slices((images_val,masks_val)).batch(BATCH_SIZE)
test_data=tf.data.Dataset.from_tensor_slices((images_test,masks_test)).batch(BATCH_SIZE)

#Defining,compiling and learning model.
EPOCHS=100
model=segmentation_model(img_size,n_channels)
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=1e-2),loss='binary_crossentropy',metrics=[Dice,IoU])
es=tf.keras.callbacks.EarlyStopping(monitor='val_Dice',min_delta=0,patience=2,restore_best_weights=True,mode='max')
history=model.fit(train_data,validation_data=val_data,epochs=EPOCHS,callbacks=[es])

#Evaluating model on a test data.
model.evaluate(test_data)

#Printing graphics of loss and metrics during learning. 
epochs=[i for i in range(1,len(history.history['loss'])+1)]
loss=history.history['loss']
val_loss=history.history['val_loss']
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_title('Loss')
ax.plot(epochs,loss,color='r',label='Train')
ax.plot(epochs,val_loss,color='b',label='Val')
ax.legend()
ax.grid()
plt.show()

epochs=[i for i in range(1,len(history.history['IoU'])+1)]
IoU=history.history['IoU']
val_IoU=history.history['val_IoU']
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_title('IoU')
ax.plot(epochs,IoU,color='g',label='Train')
ax.plot(epochs,val_IoU,color='m',label='Val')
ax.legend()
ax.grid()
plt.show()

epochs=[i for i in range(1,len(history.history['Dice'])+1)]
dice=history.history['Dice']
val_dice=history.history['val_Dice']
fig=plt.figure(figsize=(10,10))
ax=fig.add_subplot(1,1,1)
ax.set_title('Dice')
ax.plot(epochs,dice,label='Train')
ax.plot(epochs,val_dice,label='Val')
ax.legend()
ax.grid()
plt.show()

#Saving model(may be necessary to change model path).
model_path='model.h5'
model.save(model_path)