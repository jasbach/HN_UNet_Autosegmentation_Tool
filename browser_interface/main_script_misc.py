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
    
    TODAY = datetime.date.today()
    
    thresholds = {"BrainStem":0.2, "CochleaL":0.2, "CochleaR":0.2,
                  "ParotidL":0.33, "ParotidR":0.33, "SubmandibularL":0.1,
                  "SubmandibularR":0.1,"BrachialPlexus":0.1, "Brain":0.66,
                  "Larynx":0.5, "SpinalCord":0.1}
    
    region = "Head and Neck"
    filters = {"bone":[2000, 400], "tissue":[400,40],"none":[4500,1000]}
    #CRITICAL THING TO LEARN
    # how does JavaScript interact with Python to pass the image folder to the script?
    # possible that JavaScript may need to load the files first, then pass the files themselves to Python - adjust scripts if necessary
    
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
    #CoronalModel = model.get_unet(image_size)
    #SagittalModel = model.get_unet(image_size)
    
    structuresetdata = []
    
    AxialInput = np.copy(inputarray)
    
    """
    commenting out mock-3D approach, processing time is too long by a lot.
    would love to revisit 3D approaches in the future but will require more elegance
    
    original_size = len(AxialInput)
    CoronalInput = np.copy(inputarray)
    if len(CoronalInput) > image_size:
        CoronalInput = CoronalInput[:image_size]
    CoronalInput = image_prep.vertical_pad(CoronalInput,image_size)
    CoronalInput = np.moveaxis(CoronalInput,1,0)
    SagittalInput = np.copy(inputarray)
    if len(SagittalInput) > image_size:
        SagittalInput = SagittalInput[:image_size]
    SagittalInput = image_prep.vertical_pad(SagittalInput,image_size)
    SagittalInput = np.moveaxis(SagittalInput,2,0)
    """    
    
    for ROI in ROIlist:
        print("Beginning work on",ROI)
        threshold = thresholds[ROI]
        
        axialweightspath = os.path.join(weightspaths["Axial"], ROI)
        #coronalweightspath = os.path.join(weightspaths["Coronal"], ROI)
        #sagittalweightspath = os.path.join(weightspaths["Sagittal"], ROI)
        
        AxialModel.load_weights(os.path.join(axialweightspath,os.listdir(axialweightspath)[0]))
        #CoronalModel.load_weights(os.path.join(coronalweightspath,os.listdir(coronalweightspath)[0]))
        #SagittalModel.load_weights(os.path.join(sagittalweightspath,os.listdir(sagittalweightspath)[0]))
        
        if ROI == "BrachialPlexus" or ROI == "SpinalCord": #this is the spot to edit if we want to change how filters are applied
            win_lev = "Bone"
        else:
            win_lev = "Tissue"
        
        if win_lev == "Tissue":
            filtAxialInput = image_prep.apply_window_level(AxialInput,filters["tissue"][0],filters["tissue"][1])
            #filtCoronalInput = image_prep.apply_window_level(CoronalInput,filters["tissue"][0],filters["tissue"][1])
            #filtSagittalInput = image_prep.apply_window_level(SagittalInput,filters["tissue"][0],filters["tissue"][1])
        elif win_lev == "Bone":
            filtAxialInput = image_prep.apply_window_level(AxialInput,filters["bone"][0],filters["bone"][1])
            #filtCoronalInput = image_prep.apply_window_level(CoronalInput,filters["bone"][0],filters["bone"][1])
            #filtSagittalInput = image_prep.apply_window_level(SagittalInput,filters["bone"][0],filters["bone"][1])
        elif win_lev == "None":
            filtAxialInput = image_prep.apply_window_level(AxialInput,filters["none"][0],filters["none"][1])
            #filtCoronalInput = image_prep.apply_window_level(CoronalInput,filters["none"][0],filters["none"][1])
            #filtSagittalInput = image_prep.apply_window_level(SagittalInput,filters["none"][0],filters["none"][1])
        
        #AxialOutput = AxialModel.predict(filtAxialInput,verbose=0)
       
        """
        CoronalOutput = CoronalModel.predict(filtCoronalInput,verbose=0)
        SagittalOutput = SagittalModel.predict(filtSagittalInput,verbose=0)
        
        CoronalOutput = np.moveaxis(CoronalOutput,0,1)
        SagittalOutput = np.moveaxis(SagittalOutput,0,2)
        
        CoronalOutput = image_prep.unpad(CoronalOutput,original_size)
        SagittalOutput = image_prep.unpad(SagittalOutput,original_size)
        
        combinedoutput = AxialOutput*0.5 + CoronalOutput*0.25 + SagittalOutput*0.25
        """
        
        npoutputfolder = r"F:\machine learning misc\outputs\%s" % username
        if os.path.isdir(npoutputfolder) == False:
            os.mkdir(npoutputfolder)
        outputfilepath = os.path.join(npoutputfolder,"%s_predicted.npy" % ROI)
        AxialOutput = np.load(outputfilepath)
        #np.save(outputfilepath,AxialOutput)
        
        structuresetdata.append([ROI,AxialOutput,heightlist]) #if returning to 3D, change AxialOutput to combinedoutput
        
    patient_data,UIDdict = createdicomfile.gather_patient_data(imagefolder)
    structure_set = createdicomfile.create_dicom(patient_data,UIDdict,structuresetdata,image_size=256,threshold=threshold)
    
    #filename = username + "_created_ss-" + str(TODAY) + ".dcm"
    filename = "RS.%s-CNN.dcm" % username
    
    structure_set.save_as(os.path.join(outputfolder, filename), write_like_original=False)
    
if __name__ == "__main__":

    IDlist = ["017_064","017_074","017_075","017_079","017_080","017_108","017_109",
              "017_110","017_111","017_119","017_129","017_134","017_135","017_139",
              "017_140","017_141","017_143","017_144","017_145"]


    for ID in IDlist:
        imagefolder = os.path.join(r"F:\DICOMdata\RoswellData",ID)
        outputfolder = r"F:\DICOMdata\created_dicoms"
        username = ID
        patient_data,UIDdict = createdicomfile.gather_patient_data(imagefolder)
        generate_ss(imagefolder,outputfolder,username)