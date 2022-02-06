# -*- coding: utf-8 -*-
"""
@author: johna
"""

import numpy as np
import cv2
import pydicom
import os
import matplotlib.pyplot as plt

"""
Toolbox for data handling DICOM image and structure set files for use in machine learning applications.

Functions in this script:
    extract_dicom
        Input: directory path (string) or list of paths. Desired image size (pixels)
        Output: dictionaries with nested data storage
    crop_center
        Input: 2D image array, size to crop to
        Output: cropped array (cropped to center)
    pad_image
        Input: 2D image array, size to pad to
        Output: padded array (original image centered, edges padded)
    process_image - used by extract_dicom
        Input: DICOM file, image size, pixel size
        Output: 2D NumPy array of image data
    process_ss - used by extract_dicom
        Input: DICOM file
        Output: nested dictionary keyed by ROI
    build_arrays
        Input: image dictionary, label coord dictionary, name of ROI to build, image size, pixel size
        Output: whole patient CT study 3D arrays of images and segmentation masks
    visual_check
        Input: 3D arrays of images and masks (output from build_arrays works)
        Output: prints randomly selected "live slice" for alignment check
    window_level
        Input: 3D patient images array, window value, level value
        Output: Filtered 3D images array
        
Script is intended to be imported for general use. At the very end of the script
is an if __name__ == "__main__" section that you can use to validate data independently
using only this script.
"""

def extract_dicom(dir_path, image_size=512):
    #set up to handle multiple directories if needed - list or str
    if type(dir_path) == str:
        dir_list = [dir_path]
    elif type(dir_path) == list:
        dir_list = dir_path
    else: #validate input, if not list or str raise error
        raise TypeError("Incorrect variable type for directory path.")
        
    #variable dir_list is now ready - if only one directory was provided,
    #then it will be a list of lenth 1
    
    patientimagedict = {}
    patientlabeldict = {} #initialize placeholder dictionaries
    
    """
    formatting: nested dictionaries. parent dict for each initialized above
    imagedict: keys - patientID, data - daughter dictionaries
        daughter dict: keys - SliceLocation, data - 2D numpy array with image
    labeldict: keys - patientID, data - daughter dictionaries
        daughter dict: keys - ROI name, data - daughter2 dict
            daughter2 dict: keys - Slice position, data - list of x,y coords
    """
    
    for directory in dir_list:
        try:
            assert type(directory)==str
        except:
            print("A provided directory path is not in string format, it will be disregarded.")
            continue
        for root, dirs, files in os.walk(directory): #we want to extract every dicom file in directory
            for name in files:
                if name.endswith(".dcm"):
                    filepath = os.path.join(root,name)
                    dicomfile = pydicom.read_file(filepath) #will load each DICOM file in turn
                    
                    if dicomfile.Modality == "CT":
                        ID = dicomfile.PatientID
                        if ID not in patientimagedict.keys():
                            patientimagedict[ID] = {} #initializes sub-dictionary if this is first image for patient
                        singleimagearray = process_image(dicomfile, image_size)
                        sliceheight = round(dicomfile.SliceLocation * 2) / 2 #rounded to nearest 0.5
                        patientimagedict[ID][sliceheight] = singleimagearray #stores 2D numpy array of HU values for image
                        
                    if dicomfile.Modality == "RTSTRUCT":
                        ID = dicomfile.PatientID
                        structuresetdict = process_ss(dicomfile)
                        if ID in patientlabeldict.keys():
                            print("Duplicate SS for",ID)
                            raise Exception
                        patientlabeldict[ID] = structuresetdict
                                 
    return patientimagedict, patientlabeldict

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
    return image

def process_ss(file):
    
    """
    Retrieves a nested dictionary of following structure:
        ROI -> slice position -> 2D contour coords
    coordinates are in real-space (mm), unscaled
    this function does NOT create a numpy array
    for ML purposes, it's best to create the label array alongside the image array
    this isolates and extracts the necessary data to construct such an array later
    """
    structuresetdict = {}
    for ROI in file.RTROIObservationsSequence:
        structuresetdict[ROI.ROIObservationLabel] = {}
        ref_num = ROI.ReferencedROINumber
        for contour in file.ROIContourSequence:
            if contour.ReferencedROINumber == ref_num:
                for contourslice in contour.ContourSequence:
                    coords = np.asarray(contourslice.ContourData)
                    num_points = int(np.size(coords)/3)
                    coords = coords.reshape(num_points,3)
                    contour_height = round(coords[0,2] * 2) / 2
                    coords2d = np.array([coords[:,0],coords[:,1]])
                    coords2d = np.transpose(coords2d)
                    #coordinates are now prepared as a numpy array, we need to handle
                    #recording the coord array into the dictionary with accomodation for bilateral or nodal ROIs
                    if contour_height not in structuresetdict[ROI.ROIObservationLabel]:
                        structuresetdict[ROI.ROIObservationLabel][contour_height] = coords2d
                    else:
                        tempdata = structuresetdict[ROI.ROIObservationLabel][contour_height]
                        if type(tempdata) != list:
                            templist = [tempdata]
                        elif type(tempdata) == list:
                            templist = tempdata.copy()
                        templist.append(coords2d)
                        structuresetdict[ROI.ROIObservationLabel][contour_height] = templist
                    
                    #now the contour coordinates are keyed into the dictionary by slice height
                    #contour data either a np.array or a list containing multiple arrays
                    
                    #note that this function stores coordinates keyed by the ROI label in the
                    #DICOM file. these are not always standardized. consider ways to unify labeling
                    #at a later time if using for a project.
    return structuresetdict

def build_parallel_arrays(imagedict,labeldict,ROItobuild,image_size=512,pixel_size=0.5):
    """

    Parameters
    ----------
    imagedict : Dictionary
        Image data, keys are slice position and entries are 2D image arrays in HU.
    labeldict : Dictionary
        Nested dictionary of ROIlabel -> slice position -> 2D coordinates
    ROItobuild : String
        Name of the ROI to build a label array of. Must match keys in labeldict.
    image_size : int
        x,y number of pixels in image for processing. (may look to infer this later)
    pixel_size : float
        mm width of pixels, used to convert contour coordinates (in mm) to array
        positions within the segmentation mask array.

    Returns
    -------
    imagearray : NumPy array
        3D array of stacked image data
    labelarray : NumPy array
        3D array of segmentation mask, corresponding to imagearray

    """
    
    imagearray = []
    labelarray = []
    
    contourinfo = labeldict[ROItobuild]
    sliceheights = np.asarray(sorted(imagedict.keys()))
    #need to figure out a way to standardize slice thickness
    slicethickness = 99 #placeholder value
    for i in range(1,len(sliceheights)):
        if abs(sliceheights[i] - sliceheights[i-1]) < slicethickness:
            slicethickness = abs(sliceheights[i] - sliceheights[i-1])
            #sets the slice thickness to the smallest interval found so far
    if len(np.unique(np.diff(sliceheights))) > 1: #if-clause triggers when slice thickness is not standardized
        thicknesslist = np.diff(sliceheights)
        if np.any(np.unique(thicknesslist % slicethickness) != 0):
            raise ValueError("Nonstandard slice thicknesses")
        if slicethickness*2 in thicknesslist:
            newslices = sliceheights[np.argwhere(thicknesslist==(slicethickness*2))] + slicethickness
            sliceheights = np.sort(np.append(sliceheights, newslices))
            #add placeholder slice of standard size into slice list
        if slicethickness*3 in thicknesslist:
            newslices = sliceheights[np.argwhere(thicknesslist==(slicethickness*3))] + slicethickness
            alsoadd = newslices + slicethickness
            #adding two additional placeholder slices of standardized size into slice list
            sliceheights = np.sort(np.concatenate((sliceheights,newslices,alsoadd)))
        if np.any(thicknesslist >= slicethickness*4):
            raise ValueError("Slice thickness differences too large!")
            
        assert len(np.unique(np.diff(sliceheights))) == 1
        #at this point all slice thicknesses should be standardized
        #may need to streamline error handling somehow
    
    """
    REVISIT THIS: would like code it to give an alert if contour data is being "dropped"
    
    the below code was my first try, it wasn't working for some reason. need to look into it
    
    if all(item in contourinfo.keys() for item in sliceheights) is False:
        print("Contour heights exist that do not match image heights, double check input data.")
        raise ValueError
    """
    
    imagearray = np.zeros((len(sliceheights),image_size,image_size))
    labelarray = np.zeros((len(sliceheights),image_size,image_size))
    
    #at this point, our sliceheights array is already sorted, smallest to largest value
    #we'll iterate through it and fill our arrays with data

    for i in range(len(sliceheights)):
        if sliceheights[i] in imagedict:
            imagearray[i,:,:] = imagedict[sliceheights[i]]
        if sliceheights[i] in contourinfo:
            contourdata = contourinfo[sliceheights[i]]
            if type(contourdata) != list:
                contourdata = [contourdata]
            for coords in contourdata:
                corrected_coords = np.round((coords / pixel_size),decimals=0)
                corrected_coords = corrected_coords + image_size/2 #corrects origin from 0,0 to corner for use in array
                #all coords are now positive mm expressions of position. divide by pixel size to get correct location
                corrected_coords[corrected_coords < 0] = 0
                corrected_coords[corrected_coords > image_size] = image_size
                #bounds contour coordinates to image region
                maskslice = np.copy(labelarray[i,:,:])
                cv2.fillPoly(maskslice,np.int32([corrected_coords]),color=(1))
                labelarray[i,:,:] = maskslice
    

    for i in range(len(imagearray)):
        if np.sum(imagearray[i]) == 0: #at this point all slices of image array will have image data except ones created to standardize the slice thickness
            imagearray[i] = imagearray[i-1]
            labelarray[i] = labelarray[i-1]
            #for thicker slices that have been divided into multiple slices for
            #standardization purposes, duplicate the data from the "base" slice
    
    return imagearray,labelarray

def visual_check(imagearray,labelarray):
    """
    Quick function to validate your arrays. Randomly selects a slice that is "live" for contour data and displays
    image with mask overlaid.
    
    Parameters
    ----------
    imagearray : array
        3D NumPy array with HU values in pixel positions for image stack
    labelarray : array
        3D NumPy array with segmentation mask corresponding to images

    Returns
    -------
    None.

    """
    liveslices = []
    for i in range(len(labelarray)):
        if np.sum(labelarray[i]) != 0:
            liveslices.append(i)
    liveslices = np.asarray(liveslices)
    randompull = np.random.choice(liveslices)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(imagearray[randompull],cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(imagearray[randompull],cmap='gray')
    plt.imshow(labelarray[randompull],cmap='jet',alpha=0.5)
    plt.show()
    
    return

def window_level(imagearray,window,level):
    upperlimit = level + (window / 2)
    lowerlimit = level - (window / 2)
    imagearray[imagearray > upperlimit] = upperlimit
    imagearray[imagearray < lowerlimit] = lowerlimit
    return imagearray
    

if __name__ == "__main__":
    #primary intent of script is to be imported as a toolbox, but this clause
    #lets you run it independently for validation purposes
    parent_dir = "F:\\DICOM data\\RoswellData\\017_051"
    patientimagedict, patientlabeldict = extract_dicom(parent_dir)
    testimagearray,testlabelarray = build_parallel_arrays(patientimagedict['017_051'],patientlabeldict['017_051'],'Spinal Cord')