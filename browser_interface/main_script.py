# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 22:32:51 2020

@author: johna
"""

import os
import numpy as np
import datetime

import image_prep
import model
#import predict_pretrained needs rework
import createdicomfile

"""

Main script for handling dataflow for automatic contour generation for clinicians

INPUT RECEIVED FROM BROWSER:
    
    Folder with DICOM image files

"""


def generate_ss(imagefolder,outputfolder,username):
    
    thresholds = {"BrainStem":0.2, "CochleaL":0.2, "CochleaR":0.2,
                  "ParotidL":0.33, "ParotidR":0.33, "SubmandibularL":0.1,
                  "SubmandibularR":0.1,"BrachialPlexus":0.1, "Brain":0.66,
                  "Larynx":0.5, "SpinalCord":0.1}
    
    region = "Head and Neck"
    filters = {"bone":[2000, 400], "tissue":[400,40],"none":[4500,1000]}
    
    datalist = image_prep.get_list_of_datasets(imagefolder)
    inputarray, heightlist = image_prep.build_array(datalist,image_size=256,pixel_size=1)
    
    if region == "Head and Neck":
        ROIlist = ["BrachialPlexus","Brain","CochleaL","CochleaR","Larynx","ParotidL","ParotidR","SpinalCord",
                     "BrainStem","SubmandibularL","SubmandibularR"]
        
        weightspaths = {"Axial":"F:\\machine learning misc\\weights\\3D\\Axial",
                        "Coronal":"F:\\machine learning misc\\weights\\3D\\Coronal",
                        "Sagittal":"F:\\machine learning misc\\weights\\3D\\Sagittal"}
    
    image_size = 256
    AxialModel = model.get_unet(image_size)

    
    structuresetdata = []
    
    AxialInput = np.copy(inputarray)
    

    for ROI in ROIlist:
        print("Beginning work on",ROI)
        threshold = thresholds[ROI]
        
        axialweightspath = os.path.join(weightspaths["Axial"], ROI)

        
        AxialModel.load_weights(os.path.join(axialweightspath,os.listdir(axialweightspath)[0]))

        
        if ROI == "BrachialPlexus" or ROI == "SpinalCord": #this is the spot to edit if we want to change how filters are applied
            win_lev = "Bone"
        else:
            win_lev = "Tissue"
        
        if win_lev == "Tissue":
            filtAxialInput = image_prep.apply_window_level(AxialInput,filters["tissue"][0],filters["tissue"][1])

        elif win_lev == "Bone":
            filtAxialInput = image_prep.apply_window_level(AxialInput,filters["bone"][0],filters["bone"][1])

        elif win_lev == "None":
            filtAxialInput = image_prep.apply_window_level(AxialInput,filters["none"][0],filters["none"][1])

        
        AxialOutput = AxialModel.predict(filtAxialInput,verbose=0)
        

        structuresetdata.append([ROI,AxialOutput,heightlist]) #if returning to 3D, change AxialOutput to combinedoutput
        
    patient_data,UIDdict = createdicomfile.gather_patient_data(imagefolder)
    structure_set = createdicomfile.create_dicom(patient_data,UIDdict,structuresetdata,image_size=256,threshold=threshold)
    
    filename = "RS.%s-CNN.dcm" % username
    
    structure_set.save_as(os.path.join(outputfolder, filename), write_like_original=False)
    