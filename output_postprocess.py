# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 23:29:01 2020

@author: johna
"""

import numpy as np
import cv2

def apply_threshold(array,threshold):
    array[array > threshold] = 1
    array[array <= threshold] = 0
    return array

def scrub_output(originalarray,bilateral=False):
    """
    Parameters
    ----------
    originalarray : 3D NumPy array
        output array from neural network
    bilateral : Boolean, optional
        indicates whether ROI is bilateral. The default is False.

    Returns
    -------
    returned_array : 3D NumPy array
        binary array with only one (or two, if bilateral) region per slice

    """
    if len(np.unique(originalarray)) != 2:
        print("WARNING: Input array not thresholded/binary.")
    
    returned_array = []
    for i in range(0,len(originalarray)):
        newmask = np.zeros((originalarray.shape[1],originalarray.shape[2]),dtype='uint8')          
        contours, heirarchy = cv2.findContours(originalarray[i].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        #findContours function outputs a list of arrays
        contours = np.array(sorted(contours, key=len, reverse=True))
        if len(contours) > 0:
            contour_coords = contours[0]
            contour_coords = np.squeeze(contour_coords)
            for point in contour_coords:
                point = np.squeeze(point)
            contour_coords = contour_coords.astype('int32')
            if len(contour_coords) > 2:
                newmask = cv2.fillPoly(newmask, pts =[contour_coords], color=(1))
            #add clause for bilateral organs to repeat for the second largest single region
            if bilateral == True and len(contours) > 1:
                contour_coords = contours[1]
                contour_coords = np.squeeze(contour_coords)
                for point in contour_coords:
                    point = np.squeeze(point)
                contour_coords = contour_coords.astype('int32')
                if len(contour_coords) > 2:
                    newmask = cv2.fillPoly(newmask, pts =[contour_coords], color=(1))
        returned_array.append(newmask)
    
    returned_array = np.asarray(returned_array)
    return returned_array

def simple_z_smoothing(returned_array):
    smoothed_array = np.copy(returned_array)
    for i in range(1,len(returned_array)-1):
        interpolate = (returned_array[i-1,:,:] == returned_array[i+1,:,:])  
            #creates 2D boolean array of slice named "interpolate" to feed to np.where as interpolation guide
            #assigns true values when value of position above and position below match
        smoothed_array[i,:,:] = np.where(interpolate, 
                                         (returned_array[i-1,:,:]+returned_array[i+1,:,:])/2,
                                         returned_array[i,:,:])
        #np.where uses interpolate as an activation filter - if true, it sets value of position of interest to equal
                #the value of the positions above and below, whether 1 or 0
    return smoothed_array

def height_prior(binary_array,maxheight):
    compressed = np.zeros(len(binary_array))
    for i in range(len(binary_array)):
        compressed[i] = np.sum(binary_array[i])
    index = 0
    maxwindow = 0
    maxindex = 0
    while index+maxheight < len(compressed):
        if np.sum(compressed[index:index+maxheight]) > maxwindow:
            maxwindow = np.sum(compressed[index:index+maxheight])
            maxindex = index
        index += 1
    bound_array = np.zeros(binary_array.shape)
    bound_array[maxindex:maxindex+maxheight] = binary_array[maxindex:maxindex+maxheight]
    return bound_array

def scrap_stray(binary_array):
    compressed = np.zeros(len(binary_array))
    for i in range(len(binary_array)):
        compressed[i] = np.sum(binary_array[i])
    idx = np.squeeze(np.argwhere(compressed))
    chunks = np.split(compressed[idx],np.where(np.diff(idx)!=1)[0]+1)
    chunkvals = np.zeros(len(chunks))
    for i in range(len(chunks)):
        chunkvals[i] = np.sum(chunks[i])
    biggest = chunks[np.argmax(chunkvals)]
    N = len(biggest)
    possibles = np.where(compressed == biggest[0])[0]
    for p in possibles:
        if np.all(compressed[p:p+N] == biggest):
            startindex = p
    clean_array = np.zeros(binary_array.shape)
    clean_array[startindex:startindex+N] = binary_array[startindex:startindex+N]
    return clean_array
        