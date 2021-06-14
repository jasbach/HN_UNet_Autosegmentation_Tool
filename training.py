# -*- coding: utf-8 -*-
"""
Created on Tue May 25 23:14:08 2021

@author: johna
"""

import os
import sys
import math
import logging
logger = logging.getLogger(name="Training")

import keras
import tensorflow as tf
import numpy as np
from keras import optimizers
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator

import file_handling
from model.UNet import Build_UNet
from model.losses import bce_dice_loss, dice_coef

"""
Utility script for training your own UNet for image segmentation.

Note that the data must be stored in one data folder with each patient study as a subdirectory in that directory.
Each subdirectory should have every CT image file and the RTSTRUCT file to generate ground truth labels for training.
"""


#==============================================================
# User-defined variables that determine the training settings
#==============================================================
CONTOUR = "Larynx"
MAINROOTPATH = "F:/DICOMdata/RoswellData" #Path to the root data directory for training data 

VAL_SPLIT = 0.2
RANDOM_VAL = True #if False, takes first X patients as validation set. if True, selects patients randomly for val set

IMAGE_SIZE = 256
PIXEL_SIZE = 1.0 #current version not equipped for varying pixel size

#==============================================================
#     Prepare training data for feed into neural network
#==============================================================
patients_list = []
for item in os.listdir(MAINROOTPATH):
    if os.path.isdir(os.path.join(MAINROOTPATH,item)):
        patients_list.append(item) #assumption is that all subdirectories in root directory represent a patient

if RANDOM_VAL == False:
    val_patients = patients_list[0:int(len(patients_list)*VAL_SPLIT)]
elif RANDOM_VAL == True:
    val_patients = np.random.choice(patients_list,int(len(patients_list)*VAL_SPLIT),replace=False)

for item in os.listdir(MAINROOTPATH):
    if os.path.isfile(os.path.join(MAINROOTPATH,item)):
        logger.warning("Found file %s in root directory, ignoring.", item)
    if os.path.isdir(os.path.join(MAINROOTPATH,item)):
        patientpath = os.path.join(MAINROOTPATH, item)
        filelist,ss = file_handling.get_files(patientpath,get_rtstruct=True)
        valid, ref_num = file_handling.check_patient_validity(ss,CONTOUR)
        if valid == False:
            logger.warning("Patient does not have usable contour data, bypassing %s.", item)
            continue
        imagearray, heightlist = file_handling.build_array(filelist,IMAGE_SIZE,PIXEL_SIZE)
        contourpoints = file_handling.get_contour_points(ss,ref_num)
        labelarray = file_handling.build_mask(contourpoints,heightlist,IMAGE_SIZE,PIXEL_SIZE)
        
        if item not in val_patients:
            try:
                inputdata = np.concatenate((inputdata,imagearray),axis=0)
                inputlabel = np.concatenate((inputlabel,labelarray),axis=0)
            except NameError:
                inputdata = imagearray
                inputlabel = labelarray
        elif item in val_patients:
            try:
                valdata = np.concatenate((valdata,imagearray),axis=0)
                vallabel = np.concatenate((vallabel,labelarray),axis=0)
            except NameError:
                valdata = imagearray
                vallabel = labelarray

#model requires 4D array for feed data (fourth dimension is 'channels' which is 1 for our purposes)    
inputdata = np.expand_dims(inputdata,axis=-1)
inputlabel = np.expand_dims(inputlabel,axis=-1)
valdata = np.expand_dims(valdata,axis=-1)
vallabel = np.expand_dims(vallabel,axis=-1)

#==============================================================
#           Define data augmentation scheme
#==============================================================
AUG_INTENSITY = 1 #can be adjusted to add more severe data augmentation
BATCH_SIZE = 32

data_gen_args = dict(rotation_range=10 * AUG_INTENSITY,
                    width_shift_range=0.15 * AUG_INTENSITY,
                    height_shift_range=0.15 * AUG_INTENSITY,
                    shear_range=0.2 * AUG_INTENSITY,
                    zoom_range=0.1 * AUG_INTENSITY,
                    fill_mode='nearest')
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)
seed = 811

image_generator = image_datagen.flow(inputdata, batch_size=BATCH_SIZE, seed=seed)
mask_generator = mask_datagen.flow(inputlabel, batch_size=BATCH_SIZE, seed=seed)
train_generator = zip(image_generator, mask_generator)

#==============================================================
#           Build and compile the UNet itself
#==============================================================
#variables can be adjusted to modify training
NUM_EPOCHS = 200
EARLY_STOPPING_PATIENCE = 100 #stops training if no improvement over 100 epochs
LR_REDUCE_PATIENCE = 25 #triggers a reduction in learning rate if the improvement plateaus
INITIAL_LEARN_RATE = 5e-5
MINIMUM_LEARN_RATE = 1e-6
WEIGHTS_PATH = "F:/DeepLearning/weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5"

model = Build_UNet()
model.compile(optimizer=optimizers.Adam(lr=INITIAL_LEARN_RATE),
                  loss=bce_dice_loss, metrics=[dice_coef])
model_checkpoint = callbacks.ModelCheckpoint(WEIGHTS_PATH, monitor='val_loss', save_best_only=True)
model_earlystopping = callbacks.EarlyStopping(monitor='val_loss', min_delta=0,patience=EARLY_STOPPING_PATIENCE,
                                              verbose=1,mode='auto',baseline=None, restore_best_weights=True)
model_reduceLR = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.50, patience=LR_REDUCE_PATIENCE,
                                             min_delta=0.03, min_lr=MINIMUM_LEARN_RATE,verbose=1)

history = model.fit_generator(train_generator, epochs=NUM_EPOCHS, verbose=2, shuffle=True,
                    steps_per_epoch=math.ceil(len(inputdata) / BATCH_SIZE),
                    validation_data=(valdata,vallabel),
                    callbacks=[model_checkpoint, model_earlystopping, model_reduceLR])
          
final_val_loss = history.history['val_loss'][-1]

#At this point the model can be used to predict on test cases
#The best training weights will have been saved at the WEIGHTS_PATH and can be loaded using the generate DICOM script