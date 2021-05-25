# -*- coding: utf-8 -*-
"""
Created on Mon May 24 20:42:33 2021

@author: johna

This is the main script for generating a DICOM structure set file for a patient CT study for head and neck.

The only variable the user must provide is the path to the folder that holds the DICOM CT files.
"""

import os
import sys
import logging
logger = logging.getLogger(name="Main")
logger.setLevel(level="WARNING")

import keras
import tensorflow as tf
import numpy
from keras import optimizers

import file_handling
import createdicomfile
from model.UNet import Build_UNet
from model.losses import bce_dice_loss, dice_coef


patientfolder = r"F:\DICOMdata\RoswellData\017_111" #<--- Update this variable to the path to the folder that holds the patient study

if len(sys.argv) > 1:
    patientfolder = sys.argv[1] #if run in command line, allows path to folder to be passed as an argument to the script
    
if not os.path.exists(patientfolder):
    raise Exception("Invalid folder provided, re-check the input.")

#================================================
#        Initial processing of CT files
#================================================
filelist = file_handling.get_files(patientfolder)
inputarray, heightlist = file_handling.build_array(filelist)

#inputarray is now retained as the base array. For each OAR a copy will be created for the window/level filtering

#================================================
#        Initialize the model
#================================================
model = Build_UNet()
model.compile(optimizer=optimizers.Adam(lr=1e-5),
                  loss=bce_dice_loss, metrics=[dice_coef])

#================================================
#        Prepare and iterate through OARs
#================================================
ROIlist = ["BrachialPlexus","Brain","CochleaL","CochleaR","Larynx","ParotidL","ParotidR","SpinalCord",
           "BrainStem","SubmandibularL","SubmandibularR"]

filters = {"bone":[2000, 400], "tissue":[400,40],"none":[4500,1000]} #window/level filters

wd = os.getcwd()

structuresetdata = [] #This is the storage where each successive OAR array will be stored. We later can turn it into a DICOM file.
for ROI in ROIlist:
    weightspath = os.path.join(wd,"weights/%s.hdf5" % ROI)
    model.load_weights(weightspath)
    
    #apply window/level
    if any((ROI=="BrachialPlexus",ROI=="SpinalCord")):
        model_input = file_handling.apply_window_level(inputarray,filters["bone"][0],filters["bone"][1])
    else:
        model_input = file_handling.apply_window_level(inputarray,filters["tissue"][0],filters["tissue"][1])
        
    prediction = model.predict(model_input)
    
    structuresetdata.append([ROI,prediction,heightlist]) #Note that heighlist is identical for each, this is intended

#================================================
#        Generate DICOM compliant SS file
#================================================

patient_data,UIDdict = createdicomfile.gather_patient_data(filelist)
structure_set = createdicomfile.create_dicom(patient_data,UIDdict,structuresetdata,image_size=256,threshold=0.33)

filename = "%s_DeepLearning_RTSTRUCT.dcm" % patient_data["PatientID"]
structure_set.save_as(os.path.join(wd, filename), write_like_original=False) #saves to working directory