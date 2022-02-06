# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:07:46 2020

@author: johna
"""
import os
import pydicom
import numpy as np
import cv2

def get_list_of_datasets(folderpath, modality="CT"):
    filelist = []
    for root, dirs, files in os.walk(folderpath): #we want to extract every dicom file in directory
        for name in files:
            if name.endswith(".dcm"):
                filepath = os.path.join(root,name)
                dicomfile = pydicom.read_file(filepath) #will load each DICOM file in turn
                if dicomfile.Modality == modality:
                    filelist.append(dicomfile)
    return filelist

def process_image(file, image_size=512, pixel_size = 0.5):
    
    image = file.pixel_array.astype(np.int16)
    slope = file.RescaleSlope
    intercept = file.RescaleIntercept
    
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)    
    
    num_rows = file.Rows
    num_cols = file.Columns
    if num_rows != num_cols:
        raise ValueError("Image is not square!")
    scalefactor = file.PixelSpacing[0] / pixel_size
    #PixelSpacing value is the real-space width of pixels in mm
    #dividing this value by the desired pixel size gives an array scale factor for resizing
    image = cv2.resize(image,dsize=(round(num_rows*scalefactor),round(num_cols*scalefactor)))
    if image.shape[0] > image_size:
        image = crop_center(image,image_size) #if it's larger than we want, crop it to center
    elif image.shape[0] < image_size:
        image = pad_image(image,image_size) #if it's too small, pad it with -1000
    image = np.expand_dims(image,axis=2)
    return image

def apply_window_level(image, windowwidth, windowlevel, normalize = False):

    upperlimit = windowlevel + (windowwidth / 2)
    lowerlimit = windowlevel - (windowwidth / 2)
    image[image > upperlimit] = upperlimit
    image[image < lowerlimit] = lowerlimit

    if normalize == True:
        image = image.astype(np.float32)
        image = (image-lowerlimit) / (upperlimit - lowerlimit)

    return image
    
def crop_center(img,cropto):      #function used later to trim images to standardized size if too big - trims to center
    y,x = img.shape
    startx = x//2-(cropto//2)
    starty = y//2-(cropto//2)    
    return img[starty:starty+cropto,startx:startx+cropto]

def pad_image(img,image_size):    #function used to expand image to standardized size if too small - pads with -1000, HU value of air
    newimage = np.full((image_size,image_size),-1000)
    oldsize = img.shape[0]
    padsize = round((image_size - oldsize) / 2)
    newimage[padsize:padsize+oldsize,padsize:padsize+oldsize] = img
    return newimage

def vertical_pad(array,cube_size):
    pad = int((cube_size - len(array)) / 2)
    result = np.zeros([cube_size,cube_size,cube_size, 1])
    insertHere = slice(pad,pad + len(array))
    result[insertHere] = array
    return result

def unpad(array,original_size):
    pad = int((len(array) - original_size) / 2)
    sliceHere = slice(pad, pad + original_size)
    return array[sliceHere]

def build_array(filelist,image_size=512,pixel_size=0.5):
    holding_dict = {}
    heightlist = []
    array = []
    for ds in filelist:
        image = process_image(ds,image_size,pixel_size)
        sliceheight = round(ds.SliceLocation * 2) / 2
        holding_dict[sliceheight] = image
    for height in sorted(holding_dict.keys()):
        heightlist.append(height)
        array.append(holding_dict[height])
    final_array = np.asarray(array)
    return final_array, heightlist

