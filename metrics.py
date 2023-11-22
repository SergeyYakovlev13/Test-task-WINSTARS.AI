import tensorflow as tf

#Defining function for IoU metric
def IoU(y_true,y_pred,smooth=1): 
    y_true=tf.math.round(y_true)
    y_pred=tf.math.round(y_pred)
    y_true_f=tf.keras.backend.flatten(y_true)
    y_pred_f=tf.keras.backend.flatten(y_pred)
    intersection=tf.keras.backend.sum(y_true_f*y_pred_f)
    iou=(intersection+smooth)/(tf.keras.backend.sum(y_true_f)+tf.keras.backend.sum(y_pred_f)-intersection+smooth)
    return iou

#Defining function for Dice metric
def Dice(y_true,y_pred,smooth=1): 
    y_true=tf.math.round(y_true)
    y_pred=tf.math.round(y_pred)
    y_true_f=tf.keras.backend.flatten(y_true)
    y_pred_f=tf.keras.backend.flatten(y_pred)
    intersection=tf.keras.backend.sum(y_true_f*y_pred_f)
    dice=(2*intersection+smooth)/(tf.keras.backend.sum(y_true_f)+tf.keras.backend.sum(y_pred_f)+smooth)
    return dice