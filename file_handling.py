# -*- coding: utf-8 -*-
"""
Created on Mon May 24 21:00:40 2021

@author: johna
"""

import os
import sys
import logging
logger = logging.getLogger(name="Files")

import pydicom
import numpy as np
import cv2

def get_files(folderpath):
    """
    Parameters
    ----------
    folderpath : str
        User-provided input, this defines the path to the folder that holds the CT images.

    Returns
    -------
    filelist : list
        List of loaded DICOM files which will be used to iteratively build the input array

    """
    filelist = []
    totalfiles = 0
    skippedfiles = 0
    for root, dirs, files in os.walk(folderpath): #we want to extract every dicom file in directory
        for name in files:
            if name.endswith(".dcm"):
                filepath = os.path.join(root,name)
                dicomfile = pydicom.read_file(filepath) #will load each DICOM file in turn
                if dicomfile.Modality == "CT":
                    filelist.append(dicomfile)
                else:
                    skippedfiles += 1
            else:
                skippedfiles += 1
            totalfiles += 1
    logger.info("Input folder processed. %d total files were found. %d were invalid and were not retained.",totalfiles,skippedfiles)
    return filelist

def build_array(filelist,image_size=256,pixel_size=1.0):
    """
    Parameters
    ----------
    filelist : list
        List of files from get_files function
    image_size : int, optional
        Number of pixels along each axis of the image. The default is 256.
    pixel_size : float, optional
        Size in mm of x/y dimensions of pixels. The default is 1.

    Returns
    -------
    final_array : np.array
        Numpy array of stacked images.
    heightlist : list
        List of heights that correspond positionally to the images in the final_array list

    The function creates a dictionary object to associate the z-axis SliceLocation with the image. This allows the images to be
    ordered correctly regardless of what order they were loaded in. The corresponding height list will be retained as it will be necessary
    for building the eventual DICOM structure set files - contours are stored as real-space coordinates.
    """
    holding_dict = {}
    heightlist = []
    array = []
    for ds in filelist:
        image = process_image(ds,image_size,pixel_size)
        sliceheight = round(ds.SliceLocation * 4) / 4
        holding_dict[sliceheight] = image
    for height in sorted(holding_dict.keys()):
        heightlist.append(height)
        array.append(holding_dict[height])
    final_array = np.asarray(array)
    return final_array, heightlist

def process_image(file, image_size=256, pixel_size=1.0):
    """
    Parameters
    ----------
    file : pydicom DataSet object of a CT image
        Individual file object imported previously.
    image_size : int, optional
        Size of the image to return in pixels. The default is 256.
    pixel_size : float, optional
        Pixel size to rescale image to, in mm. The default is 1.0.

    Raises
    ------
    Errors if file modality is not CT or if image dimensions are not square.

    Returns
    -------
    image : np.array
        3D array of shape (256,256,1) by default representing the image
    """
    if file.Modality != "CT":
        raise Exception("Invalid modality.")
    
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
        image = pad_image(image,image_size) #if it's too small, pad it with -1000 (happens sometimes with very small native pixel sizes)
    image = np.expand_dims(image,axis=2)
    return image

def apply_window_level(image, windowwidth, windowlevel, normalize = False):

    upperlimit = windowlevel + (windowwidth / 2)
    lowerlimit = windowlevel - (windowwidth / 2)
    image[image > upperlimit] = upperlimit
    image[image < lowerlimit] = lowerlimit

    if normalize == True: #normalizes pixel values to float between 0.0 and 1.0 - not used by default
        image = image.astype(np.float32)
        image = (image-lowerlimit) / (upperlimit - lowerlimit)

    return image
    
def crop_center(img,cropto):      #function used to trim images to standardized size if too big - trims to center
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
