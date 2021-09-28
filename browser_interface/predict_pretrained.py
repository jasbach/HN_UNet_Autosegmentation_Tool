# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 00:38:16 2020

@author: johna
"""


import numpy as np
import os

import model
import input_data_generator as inputgen
import ML_toolbox as toolbox

rootpath = "F:\\DICOM data\\RoswellData"
image_size = 256

#*****************
#KEY PARAMETERS TO CHANGE

contourtofind = ["BrachialPlexus","Brain","CochleaL","CochleaR","Larynx","ParotidL","ParotidR","SpinalCord",
                 "BrainStem","SubmandibularL","SubmandibularR"]

testpatientlist = ["017_064", "017_074","017_075","017_079","017_080", "017_108",
                   "017_109","017_110","017_111","017_119","017_129","017_130",
                   "017_134","017_135","017_139","017_140","017_141","017_143",
                   "017_144","017_145"]

#*****************

for ROI in contourtofind:
    
    print("Beginning process for", ROI)
    
    if ROI == "SpinalCord":
        win_lev = "Bone"
    else:
        win_lev = "Tissue"
    
    weightspaths = {"Axial":"F:\\machine learning misc\\weights\\3D\\Axial\\%s" % ROI,
                    "Coronal":"F:\\machine learning misc\\weights\\3D\\Coronal\\%s" % ROI,
                    "Sagittal":"F:\\machine learning misc\\weights\\3D\\Sagittal\\%s" % ROI}

    outputpath = "F:\\machine learning misc\\outputs\\local\\3D\\%s" % ROI
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
        
    truemaskpath = "F:\\machine learning misc\\truemasks\\local\\3D\\%s" % ROI
    if not os.path.exists(truemaskpath):
        os.makedirs(truemaskpath)


    testimagearrays = []
    testlabelarrays = []
    for patient in testpatientlist:
        patientpath = os.path.join(rootpath,patient)
        imagearray, labelarray, heightlist = inputgen.generate_input_data(patientpath, [ROI], 
                                                                                    image_size, 1, win_lev)   #test set null rate always equals 1 to simulate real-world
        testimagearrays.append(imagearray)     #generates lists of arrays with data segmented by patient
        testlabelarrays.append(labelarray)
    
    AxialModel = model.get_unet(image_size)
    AxialModel.load_weights(os.path.join(weightspaths["Axial"],os.listdir(weightspaths["Axial"])[0]))
    CoronalModel = model.get_unet(image_size)
    CoronalModel.load_weights(os.path.join(weightspaths["Coronal"],os.listdir(weightspaths["Coronal"])[0]))
    SagittalModel = model.get_unet(image_size)
    SagittalModel.load_weights(os.path.join(weightspaths["Sagittal"],os.listdir(weightspaths["Sagittal"])[0]))
    
    for i in range(0,len(testpatientlist)):
        AxialInput = np.copy(testimagearrays[i])
        original_size = len(AxialInput)
        CoronalInput = np.copy(testimagearrays[i])
        if len(CoronalInput) > 256:
            CoronalInput = CoronalInput[:256]
        CoronalInput = toolbox.vertical_pad(CoronalInput,256)
        CoronalInput = np.moveaxis(CoronalInput,1,0)
        SagittalInput = np.copy(testimagearrays[i])
        if len(SagittalInput) > 256:
            SagittalInput = SagittalInput[:256]
        SagittalInput = toolbox.vertical_pad(SagittalInput,256)
        SagittalInput = np.moveaxis(SagittalInput,2,0)

        
        AxialOutput = AxialModel.predict(AxialInput,verbose=0)
        CoronalOutput = CoronalModel.predict(CoronalInput,verbose=0)
        SagittalOutput = SagittalModel.predict(SagittalInput,verbose=0)
        
        CoronalOutput = np.moveaxis(CoronalOutput,0,1)
        SagittalOutput = np.moveaxis(SagittalOutput,0,2)
        
        CoronalOutput = toolbox.unpad(CoronalOutput,original_size)
        SagittalOutput = toolbox.unpad(SagittalOutput,original_size)
        
        np.save(outputpath + "\\Axial_prediction_%s.npy" % testpatientlist[i],AxialOutput)
        np.save(outputpath + "\\Coronal_prediction_%s.npy" % testpatientlist[i],CoronalOutput)
        np.save(outputpath + "\\Sagittal_prediction_%s.npy" % testpatientlist[i],SagittalOutput)
        np.save(truemaskpath + "\\groundtruth_%s.npy" % testpatientlist[i], testlabelarrays[i])
        