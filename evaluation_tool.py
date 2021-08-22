import numpy as np
import cv2
import pydicom
import math
import pandas as pd
import sys
import os

from functools import reduce

"""
Script to define get_array function which retrieves a Structure Set file and
returns a 3D volume mask array to represent the chosen ROI.

At the bottom of the script is an example of how it could be run. If you want
to validate the script against your Structure Set file, replace the variables
"testfile" and "testROI" with your values and run the script.

The function can, of course, also be imported into other Python scripts and
used to generate NumPy arrays of organs for other purposes.
"""


def get_array(filepath, ROI_name, region_size = 400,
              pixel_size = 1.0, z_range=500, return_ST=False):
    """
    Parameters
    ----------
    filepath : STRING
        Full path for DICOM Structure Set file to pull contour data from.
        
    ROI_name : STRING
        OAR to pull contour data for - must be EXACT MATCH to how the OAR is
        labeled in the structure set file.
    
    region_size : INTEGER (representing mm) Default is 400.
        Contour data is stored around a (0,0) centerpoint - to store in an array,
        all coordinates must be positive. If pairing with image arrays, must match their
        size or will be misaligned. If only comparing ROI volumes, any number
        large enough to encompass the whole ROI will work. Smaller value will
        mean faster computation time.

    pixel_size : float (representing mm) Default is 1.0
        Pixel dimensions - function will round contour coordinates to nearest
        appropriate pixel. Smaller pixel size (higher resolution) will result
        in larger array and longer computations.
        
    z_range : INTEGER
        Arbitrary height of empty array space to be filled by contour data. Must
        be large enough to fit volume. Centers around z_coord of 0.
    
    return_ST : bool
        Option to return the derived slice thickness in addition to the array.
        
    Returns
    -------
    NumPy 3D array mask of OAR.
    (Optional) Derived slice thickness in pixel-equivalence

    """
    scale_factor = int(round(1 / pixel_size, 0))
    if pixel_size != (1 / scale_factor):
        print("Pixel size set from",pixel_size,"to:", 1 / scale_factor)
    
    ss = pydicom.read_file(filepath)
    
    for item in ss.StructureSetROISequence:
        if item.ROIName == ROI_name:
            ROI_ref_num = item.ROINumber
            #DICOM file does not store contour data and contour name in the same location
            #They are tied together via a reference number, which this loop retrieves
    
    contours = ss.ROIContourSequence

    mask_array = np.zeros([z_range,region_size * scale_factor, region_size * scale_factor], dtype=np.int8)

    try:
        ROI_ref_num
    except NameError:
        print("Contour not found, please check ROI name provided against structure set file.")
    else:
        for ROI in contours:
            if ROI.ReferencedROINumber == ROI_ref_num:
                zlist = []
                for element in ROI.ContourSequence:
                    z_pos = element.ContourData[2]
                    zlist.append(z_pos)
                #used to determine slice thickness
                zarray = np.sort(np.asarray(zlist))
                zarray = zarray - np.amin(zarray) #sets lowest value to 0 for simpler math
                slice_thickness = zarray[1]
                ST_val = False
                while ST_val == False:
                    if slice_thickness == 0:  #if clause to avoid divide by zero errors
                        zarray = np.delete(zarray,0)
                        zarray = zarray - np.amin(zarray)
                        slice_thickness = zarray[1]
                    ST_val = True
                    for z in zarray:
                        if z % slice_thickness != 0:
                            #if we find a value that doesn't cleanly divide,
                            #we have the wrong slice thickness, and our
                            #lowest slice is an outlier. remove that z-position
                            #and rerun the slice thickness finder lines of code
                            ST_val = False
                            zarray = np.delete(zarray,0)
                            zarray = zarray - np.amin(zarray)
                            slice_thickness = zarray[1]
                            break
                            
                
                for element in ROI.ContourSequence:
                    contour_slice = element.ContourData
                    z_pos = contour_slice[2] #retrives height of element
                    slice_index = int((z_pos/slice_thickness) + (z_range/2))
                    #contour objects are separated by slice, stored as 1D lists
                    contour_count = int(np.size(contour_slice)/3)
                    #total number of x,y,z contour points
                    contour_slice = np.asarray(contour_slice)
                    contour = contour_slice.reshape(contour_count,3)
                    
                    contour[:,0] = (contour[:,0] + (region_size)/2)
                    contour[:,1] = (contour[:,1] + (region_size)/2)
                    #shifts x and y contour coords so all are positive
                    contour_coords = np.asarray([contour[:,0],contour[:,1]])      
                    contour_coords = np.transpose(contour_coords) * scale_factor
                    contour_coords = np.round(contour_coords, decimals=0)
                    contour_coords = np.asarray(contour_coords, dtype='int32')
                    mask_array[slice_index] = cv2.fillPoly(mask_array[slice_index],pts=[contour_coords], color=(1))

                break
    
    if return_ST == True:
        return mask_array, slice_thickness
    elif return_ST == False:
        return mask_array
    
def output_postprocess(CNNoutput, bilateral=False, threshold=0.33, smoothing = True):
    """
    The postprocess function takes the CNN output, applies the binarization threshold, then operates find contour into fillpoly
    to generate a new volume. Output will be that each slice of the volume contains at most one continous area of mask.

    
    Parameters
    ----------
    CNNoutput : np.array
        Raw output from neural network.
    bilateral : bool
        Setting for bilateral organs that are stored/processed as one (for instance, Brachial Plexus)
    threshold : float
        Pixel confidence threshold for binarization. Default is 0.33.
    smoothing : bool
        Setting to use simple z-smoothing. Default True.
        
    Returns 
    -------
    returned_array : np.array
        Processed, binarized array of organ.
    """
    CNNoutput = np.squeeze(CNNoutput)
    if len(CNNoutput.shape) != 3:
        raise Exception("Invalid array shape provided. Function does not support multichannel arrays.")
    
    CNNoutput[CNNoutput > threshold] = 1
    CNNoutput[CNNoutput <= threshold] = 0
    
    returned_array = []
    for i in range(0,len(CNNoutput)):
        newmask = np.zeros((CNNoutput.shape[1],CNNoutput.shape[2]),dtype='uint8')          
        contours, heirarchy = cv2.findContours(CNNoutput[i].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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
        
    if smoothing == True:
        #smoothing function examines each z-column and corrects and XOX to XXX
        for i in range(1,len(returned_array)-1):
            interpolate = (returned_array[i-1,:,:] == returned_array[i+1,:,:])  
                #creates 2D boolean array of slice named "interpolate" to feed to np.where as interpolation guide
                #assigns true values when value of position above and position below match
            returned_array[i,:,:] = np.where(interpolate, 
                                             (returned_array[i-1,:,:]+returned_array[i+1,:,:])/2,
                                             returned_array[i,:,:])
            #np.where uses interpolate as an activation filter - if true, it sets value of position of interest to equal
                #the value of the positions above and below, whether 1 or 0
        
    return returned_array

def get_DSC(volA,volB):
    """
    Parameters
    ----------
    volA : NumPy array
    volB : NumPy array
        Two arrays to be compared. Must have same dimensions and be only 1s and
        0s, where 1s represent the volume.
        
    Returns
    -------
    Float value for Dice Similarity Coefficient (between 0.0 and 1.0)

    """

    assert volA.shape == volB.shape
    
    smooth = 1.
    volA_f = volA.flatten()
    volB_f = volB.flatten()
    intersection = np.sum(volA_f * volB_f)
    dicescore = (2. * intersection + smooth) / (np.sum(volA_f) + np.sum(volB_f) + smooth)
    return dicescore

def sensitivity(truevol, comparevol):
    """
    Parameters
    ----------
    truevol : NumPy array
        Ground truth volume.
    comparevol : NumPy array
        Volume whose sensitivity is being measured against the ground truth.
        
    Returns
    -------
    Float value for sensitivity (TPF).
    """

    assert truevol.shape == comparevol.shape
    
    truevol_f = truevol.flatten()
    comparevol_f = comparevol.flatten()
    true_positive = np.sum(truevol_f * comparevol_f)
    all_positive = np.sum(truevol_f)
    TPF = true_positive / all_positive
    return TPF

def specificity(truevol, comparevol):
    """
    Parameters
    ----------
    truevol : NumPy array
        Ground truth volume.
    comparevol : NumPy array
        Volume whose specificity is being measured against the ground truth.
        
    Returns
    -------
    Float value for specificity (TNF).
    """

    assert truevol.shape == comparevol.shape
    
    truevol_f = truevol.flatten()
    comparevol_f = comparevol.flatten()
    any_pos = truevol_f + comparevol_f
    any_pos[any_pos == 2] = 1 
    true_negative = len(truevol_f) - np.sum(any_pos) 
    #find total number of true negatives by subtracting any positive from the total number of positions
    total_neg = len(truevol_f) - np.sum(truevol_f)
    TNF = true_negative / total_neg
    return TNF

def get_distance_sets(volA, volB, slicethickness, bilateral=False):
    """
    Parameters
    ----------
    volA : NumPy array
    volB : NumPy array
        Two volumes whose surface distance is being compared. Formatted as
        [z,x,y] axis mapping.
    slicethickness : Float
        Value of width of z-axis in real space RELATIVE to value of x/y axis.
        Function not equipped to deal with x and y axis distance increments 
        that are not the same. This variable must be a number that represents
        NUMBER OF PIXELS that would "fit" between slices in z-direction. If it's
        2.5mm of real space, but pixel size is set to 0.5mm, then slice thickness
        is 5.
        
    Returns
    -------
    AtoB_distances : list
    BtoA_distances : list
        Two lists are returned, A-to-B and B-to-A (in that order). Lists contain
        each point's shortest straight-line distance to the other surface.
        Usually evaluation metrics (Hausdorff Distance, Mean Surface Distance)
        will evaluate both sets of distances combined. Units for distances in
        the output are whatever the pixel size is.

    """
    
    #this function used interally to this script to support MSD and HD functions
    
    assert volA.shape == volB.shape
    
    point_listA = []
    for i in range(0,len(volA)):
        if np.sum(volA[i]) != 0:
            contours, heirarchy = cv2.findContours(volA[i].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=len, reverse=True)
            if len(contours) > 0:
                contour_coords = contours[0]
                contour_coords = np.squeeze(contour_coords)
                for point in contour_coords:
                    point = np.squeeze(point)
                    try:
                        point[0]
                        point[1]
                    except IndexError:
                        continue
                    point_listA.append([point[0],point[1], i * slicethickness])
                if bilateral == True and len(contours) > 1:
                    contour_coords2 = contours[1]
                    contour_coords2 = np.squeeze(contour_coords2)
                    for point in contour_coords2:
                        point = np.squeeze(point)
                        try:
                            point[0]
                            point[1]
                        except IndexError:
                            continue
                        point_listA.append([point[0],point[1], i * slicethickness])
    point_arrayA = np.asarray(point_listA)
                    
    point_listB = []
    for i in range(0,len(volB)):
        if np.sum(volB[i]) != 0:
            contours, heirarchy = cv2.findContours(volB[i].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=len, reverse=True)
            if len(contours) > 0:
                contour_coords = contours[0]
                contour_coords = np.squeeze(contour_coords)
                for point in contour_coords:
                    point = np.squeeze(point)
                    try:
                        point[0]
                        point[1]
                    except IndexError:
                        continue
                    point_listB.append([point[0],point[1], i * slicethickness])
                if bilateral == True and len(contours) > 1:
                    contour_coords2 = contours[1]
                    contour_coords2 = np.squeeze(contour_coords2)
                    for point in contour_coords2:
                        point = np.squeeze(point)
                        try:
                            point[0]
                            point[1]
                        except IndexError:
                            continue
                        point_listB.append([point[0],point[1], i * slicethickness])
    point_arrayB = np.asarray(point_listB)
    
    """
    previous two loops collect a list of 3D coordinate values for every surface
    point on each volume. final result is array of shape N,3 where N is total
    number of points, each point is saved as X,Y,Z coordinates.
                    
    create list of each point's shortest straight line distance to the other
    volume's surface by checking each point's distance to each other point and
    retaining the smallest value (shortest distance)
    """
    
    AtoB_distances = []
    
    for pointA in point_arrayA:
        single_point_distances = []
        checkrange = 0
        while len(single_point_distances) == 0:
            checkrange += 5
            x_max, x_min = (pointA[0] + checkrange), (pointA[0] - checkrange)
            y_max, y_min = (pointA[1] + checkrange), (pointA[1] - checkrange)
            z_max, z_min = (pointA[2] + checkrange), (pointA[2] - checkrange)
            
            x_indx = np.where(np.logical_and(point_arrayB[:,0] <= x_max,
                                             point_arrayB[:,0] >= x_min))
            y_indx = np.where(np.logical_and(point_arrayB[:,1] <= y_max,
                                             point_arrayB[:,1] >= y_min))
            z_indx = np.where(np.logical_and(point_arrayB[:,2] <= z_max,
                                             point_arrayB[:,2] >= z_min))
            live_points = reduce(np.intersect1d,(x_indx,y_indx,z_indx))
            
            for index in live_points:
                x_diff = abs(point_arrayB[index,0] - pointA[0])
                y_diff = abs(point_arrayB[index,1] - pointA[1])
                z_diff = abs(point_arrayB[index,2] - pointA[2])
                distance = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
                single_point_distances.append(distance)
        
        AtoB_distances.append(sorted(single_point_distances)[0])

    """    
    end result is list AtoB_distances, equal in length to the number of points
    in surface A, where each list entry represents a single point's shortest
    distance to the surface B
    """

    BtoA_distances = []
    for pointB in point_arrayB:
        single_point_distances = []
        checkrange = 0
        while len(single_point_distances) == 0:
            checkrange += 5
            x_max, x_min = (pointB[0] + checkrange), (pointB[0] - checkrange)
            y_max, y_min = (pointB[1] + checkrange), (pointB[1] - checkrange)
            z_max, z_min = (pointB[2] + checkrange), (pointB[2] - checkrange)
            
            x_indx = np.where(np.logical_and(point_arrayA[:,0] <= x_max,
                                             point_arrayA[:,0] >= x_min))
            y_indx = np.where(np.logical_and(point_arrayA[:,1] <= y_max,
                                             point_arrayA[:,1] >= y_min))
            z_indx = np.where(np.logical_and(point_arrayA[:,2] <= z_max,
                                             point_arrayA[:,2] >= z_min))
            live_points = reduce(np.intersect1d,(x_indx,y_indx,z_indx))
            
            for index in live_points:
                x_diff = abs(point_arrayA[index,0] - pointB[0])
                y_diff = abs(point_arrayA[index,1] - pointB[1])
                z_diff = abs(point_arrayA[index,2] - pointB[2])
                distance = math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
                single_point_distances.append(distance)
        
        BtoA_distances.append(sorted(single_point_distances)[0])
 
    #same loop but for points in surface B to surface of volume A

    return AtoB_distances, BtoA_distances
    
def MSD_w_distanceset(AtoB, BtoA,pixel_size):
    all_distances = AtoB + BtoA
    mean_distance = np.sum(np.asarray(all_distances)) / len(all_distances)
    mean_distance = mean_distance * pixel_size
    
    return mean_distance
    
def mean_surface_distance(volA, volB,bilateral=False, pixel_size = 1, slicethickness=2.5):
    """
    Parameters
    ----------
    volA : NumPy array
    volB : NumPy array
        Two volumes whose surface distance is being compared. Formatted as
        [z,x,y] axis mapping.
    pixel_size : float
        Defaults to 1.0. Represented in mm. If your array has pixel size that
        differs from 1mm x 1mm, provide value to function so that the output
        is correct in mm.
    slicethickness : float
        Value of width of z-axis in real space relative to value of x/y axis.
        Function not equipped to deal with x and y axis distance increments 
        that are not the same. Defaults to 2.5mm as that's the most common
        value I've seen in CT studies.
        
    Returns
    ----------
    mean_distance : float
        Number value of mean surface distance between the two volume surfaces.
    """
    AtoB, BtoA = get_distance_sets(volA,volB,slicethickness/pixel_size,bilateral)
    mean_distance = MSD_w_distanceset(AtoB,BtoA,pixel_size)
    
    return mean_distance

def HD_w_distanceset(AtoB,BtoA,pixel_size,percentile=95):
    all_distances = AtoB + BtoA
    all_distances.sort()
    cutoff_index = int(len(all_distances) * (percentile/100))
    if percentile == 100:
        cutoff_index -= 1 #required for index to not go out of range at max
    hausdorff_distance = all_distances[cutoff_index] * pixel_size
    return hausdorff_distance

def hausdorff_distance(volA,volB,bilateral=False,pixel_size=1.0,slicethickness=2.5,percentile=95):
    
    """
    Parameters
    ----------
    volA : NumPy array
    volB : NumPy array
        Two volumes whose surface distance is being compared. Formatted as
        [z,x,y] axis mapping.
    percentile : integer
        Default is 95. For maximal Hausdorff Distance, set to 100
        (true definition of Hausdorff). Many studies prefer 95th percentile
        Hausdorff distance to reject outliers. Can be used for any percentile
        range; for instance, providing 50 as percentile value returns median
        distance.
    pixel_size : float
        Defaults to 1.0. Represented in mm. If your array has pixel size that
        differs from 1mm x 1mm, provide value to function so that the output
        is correct in mm.
    slicethickness : float
        Value of width of z-axis in real space relative to value of x/y axis.
        Defaults to 2.5mm as that's the most common value I've seen in CT studies.

    Returns
    -------
    hausdorff_distance : float
        Value of Hausdorff Distance for two surfaces.
    """
    
    AtoB, BtoA = get_distance_sets(volA,volB,(slicethickness/pixel_size),bilateral)
    hausdorff_distance = HD_w_distanceset(AtoB,BtoA,pixel_size,percentile=95)
    
    return hausdorff_distance
    
def full_eval(volA,volB,bilateral,pixel_size=1.0,slicethickness=2.5):
    """
    Meta-function that receives two volumes for comparison and assesses all
    three metrics.
    
    Parameters
    ----------
    volA : np.array
    volB : np.array
        Matching shape 3D arrays to be compared. Must be binary arrays where
        1 represents a voxel that is in the volume and 0 represents a voxel that
        is not.
    bilateral : bool
        Indicator of whether a bilateral organ (2 volumes) is being considered.
    pixel_size : float
        Pixel size in x/y dimensions in mm.
    slicethickness : float
        Slice thickness (z-axis) in mm
        
    Returns
    -------
    DSC, MSD, HD : tuple of floats
        Results of each metric
    """
    
    AtoB, BtoA = get_distance_sets(volA,volB,(slicethickness/pixel_size),bilateral)
    MSD = MSD_w_distanceset(AtoB,BtoA,pixel_size)
    HD = HD_w_distanceset(AtoB,BtoA,pixel_size)
    DSC = get_DSC(volA,volB)
    return DSC,MSD,HD
    
if __name__ == "__main__":
    pass
    #uncomment below and replace with your filepaths to your DICOM files
    
    # ROI_name = "Parotid L"
    # ground_truth_DICOM = "exampleGTfile.dcm"
    # prediction_DICOM = "examplePRDfile.dcm"

    # ground_truth = get_array(ground_truth_DICOM,ROI_name)
    # predict = get_array(prediction_DICOM,ROI_name)
    
    # DSC,MSD,HD = full_eval(ground_truth,predict,bilateral=False)
    # print("The predicted volume scores for DSC, MSD (mm), and HD (mm), respectively:", DSC, MSD, HD)