# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 00:32:25 2020

@author: johna
"""


#!/usr/bin/env python
# coding: utf-8

import numpy as np

from keras import models
from keras.layers import *
from keras import optimizers
from keras import callbacks
from keras import backend as K
from keras import metrics
from losses import bce_dice_loss


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dicescore = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dicescore

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def get_dice_sum(predictions, test_labels,thresholding = False):
    smooth = 1.
    dicesum = 0
    intersectionsum = 0
    denominatorsum = 0
    for i in range(0, len(predictions)):
        smooth = 1.
        
        y_true = test_labels[i][:,:,0]
        y_pred = predictions[i][:,:,0]
        if thresholding == True:
            y_pred[y_pred > 0.33] = 1
            y_pred[y_pred <= 0.33] = 0
        
        y_true_f = y_true.flatten()
        y_pred_f = y_pred.flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        intersectionsum += intersection
        denominatorsum += (np.sum(y_true_f)+np.sum(y_pred_f))
    
    dicetotal = (2. * intersectionsum + smooth) / (denominatorsum + smooth)
    return dicetotal

def get_unet(image_size, num_channels=1):   #have adjusted filter size numbers to account for image size of 256
    
    assert image_size == 256
    
    inputs = Input((256, 256, num_channels))    #third number is the number of channels - configured for single W/L
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)    

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)    

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = models.Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer=optimizers.Adam(lr=1e-5),
                  loss=bce_dice_loss, metrics=[dice_coef])  #loss options: dice_coef_loss, bce_dice_loss, weighted_dice_loss, weighted_bce_loss

    return model

def get_cnet(num_channels=1):
    outer_input = Input((256,256,num_channels))
    conv1_1 = Conv2D(64,3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(outer_input)
    conv1_2 = Conv2D(64,3,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1_2)
    conv2_1 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool1)
    conv2_2 = Conv2D(128,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2_2)
    conv3_1 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool2)
    conv3_2 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2,2))(conv3_2)
    conv4_1 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(pool3)
    conv4_2 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv4_1)
    
    OuterNetwork = models.Model(outer_input,conv4_2,name="OuterNetwork")
    
    main_input1 = Input((256,256,num_channels))
    main_input2 = Input((256,256,num_channels))
    main_input3 = Input((256,256,num_channels))
    main_input4 = Input((256,256,num_channels))
    outer1 = OuterNetwork(main_input1)
    outer2 = OuterNetwork(main_input2)
    outer3 = OuterNetwork(main_input3)
    outer4 = OuterNetwork(main_input4)
    outer_output1 = concatenate([outer1,outer2],axis=3)
    outer_output2 = concatenate([outer3,outer4],axis=3)
    
    middle_input = Input((32,32,512))
    conv1_1 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(middle_input)
    conv1_2 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1_1)
    conv1_3 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1_2)
    conv1_4 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1_3)
    conv1x1 = Conv2D(256,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv1_4)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1x1)
    drop1 = Dropout(0.5)(pool1)
    conv2_1 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(drop1)
    conv2_2 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2_1)
    conv2_3 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2_2)
    conv2_4 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv2_3)
    conv2x1 = Conv2D(256,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv2_4)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2x1)
    drop2 = Dropout(0.5)(pool2)
    
    MiddleNetwork = models.Model(middle_input,drop2,name="MiddleNetwork")
    
    middle1 = MiddleNetwork(outer_output1)
    middle2 = MiddleNetwork(outer_output2)
    middle_output = concatenate([middle1,middle2],axis=3)
    
    conv1 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(middle_output)
    conv2 = Conv2D(256,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1)
    convx = Conv2D(256,1,activation='relu',padding='same',kernel_initializer='he_normal')(conv2)
    lastpool = MaxPooling2D(pool_size=(2,2))(convx)
    drop1 = Dropout(0.5)(lastpool)
    flattened = Flatten()(drop1)
    fc1 = Dense(1024,activation='relu')(flattened)
    drop2 = Dropout(0.5)(fc1)
    fc2 = Dense(1024,activation='relu')(drop2)
    drop3 = Dropout(0.5)(fc2)
    final_output = Dense(1,activation='sigmoid')(drop3)
    
    full_model = models.Model([main_input1,main_input2,main_input3,main_input4],final_output,name="cnet")
    
    full_model.compile(optimizer=optimizers.Adam(lr=1e-4,beta_1=0.9,beta_2=0.999,epsilon=1e-7),
                       loss='binary_crossentropy', metrics=[metrics.BinaryAccuracy()])
    
    return full_model

if __name__ == "__main__":
    model = get_cnet()
    model.summary()
