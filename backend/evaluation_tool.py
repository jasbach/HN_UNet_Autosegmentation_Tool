import numpy as np
import cv2
#import pydicom
#import os
#import sys
import math
#import matplotlib.pyplot as plt
import scipy.ndimage

"""
Receive two arrays: Ground truth and array to be evaluated. 

CNN output will be formatted as it comes out of CNN,
no data adjustment will be performed prior to passing it to this function

MIMS array will be made in a similar manner as the ground truth and compared in that way

Workflow for modifying CNN output follows this general path:
Threshold (starting out trying 0.33)
Take contours using cv2
Trim to only the longest region contour
Feed the resulting points back into FillPoly to create a mask that can be compared pixelwise


Comparison metrics to be implemented:
DICE Similarity Coefficient
Average 2D Hausdorff Distance
Average 2D Center of Mass Distance
Sensitivity and Specificity

Can implement more evaluation functions as necessary


Assume that ground truth is processed appropriately, need to write a function to post-process CNN output
"""

#dice function takes as inputs two 3D arrays - the ground truth mask volume and the volume to be compared ("predict")
def get_dicescore(truemask, predict):
    smooth = 1.
    truemask_f = truemask.flatten()
    predict_f = predict.flatten()
    intersection = np.sum(truemask_f * predict_f)
    dicescore = (2. * intersection + smooth) / (np.sum(truemask_f) + np.sum(predict_f) + smooth)
    return dicescore

def sensitivity(truemask, predict):
    truemask_f = truemask.flatten()
    predict_f = predict.flatten()
    true_positive = np.sum(truemask_f * predict_f)
    all_positive = np.sum(truemask_f)
    TPF = true_positive / all_positive
    return TPF

def specificity(truemask, predict):
    truemask_f = truemask.flatten()
    predict_f = predict.flatten()
    any_pos = truemask_f + predict_f
    any_pos[any_pos == 2] = 1 
    true_negative = len(truemask_f) - np.sum(any_pos) #find total number of true negatives by subtracting any positive from the total number of positions
    correct_neg = len(truemask_f) - np.sum(truemask_f)
    TNF = true_negative / correct_neg
    return TNF

# np.squeeze before passing to function
def output_postprocess(CNNoutput, bilateral=False, threshold=0.33, smoothing = True):    
    assert len(CNNoutput.shape) == 3
    
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
#the postprocess function takes the CNN output, applies the threshold, then operates find contour into fillpoly
    #to generate a new volume...output will be that each slice of the volume contains at most one continous area of mask
    #this function would prevent predicting multiple small regions as currently written, however
    #because the find contours function generates a list of arrays, it would be possible to
    #run a looped fillpoly function for each contour extracted
    

def COM2d_compare(truemask, predict):
    slicesindexed = 0
    diffsum = 0
    for i in range(0,len(truemask)):
        if np.sum(truemask[i]) > 0 and np.sum(predict[i]) > 0:
            truthcom = scipy.ndimage.measurements.center_of_mass(truemask[i])
            predictcom = scipy.ndimage.measurements.center_of_mass(predict[i])
            x_diff = abs(truthcom[0] - predictcom[0])
            y_diff = abs(truthcom[1] - predictcom[1])
            vector_diff = math.sqrt(x_diff**2 + y_diff**2)
            diffsum += vector_diff
            slicesindexed += 1
    average_diff = diffsum / slicesindexed
    return average_diff

def mean_surface_distance(truemask, predict, bilateral = False):
    total_surface_distance = 0
    sliceschecked = 0
    for image in range(0,len(truemask)):
        if np.sum(truemask[image]) == 0:
            continue
        truecontours, heirarchy = cv2.findContours(truemask[image].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        truecontours = np.array(sorted(truecontours, key=len, reverse=True))
        truecontour_coords = np.asarray(truecontours[0])
        truecontour_coords = np.squeeze(truecontour_coords)
        #add clause for bilateral organs to repeat for the second largest single region
        if bilateral == True and len(truecontours) > 1:
            contour_coords = np.asarray(truecontours[1])
            contour_coords = np.squeeze(contour_coords)
            if len(contour_coords) > 2:
                truecontour_coords = np.concatenate((truecontour_coords,contour_coords), axis=0)

        if np.sum(predict[image]) == 0:
            continue
        predcontours, heirarchy = cv2.findContours(predict[image].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        predcontours = np.array(sorted(predcontours, key=len, reverse=True))
        predcontour_coords = predcontours[0]
        predcontour_coords = np.squeeze(predcontour_coords)
        if bilateral == True and len(predcontours) > 1:
            contour_coords = predcontours[1]
            contour_coords = np.squeeze(contour_coords)
            if len(contour_coords) > 2:
                predcontour_coords = np.concatenate((predcontour_coords,contour_coords), axis=0)
                
        for point in predcontour_coords:
            point = np.squeeze(point)
        for point in truecontour_coords:
            point = np.squeeze(point)

        if len(predcontour_coords.shape) < 2 or len(truecontour_coords.shape) < 2:
            continue
        
        pointschecked = 0
        distancesum = 0        
        for i in range(0,len(predcontour_coords)):
            shortest = 1000      #initialize with arbitrarily large distance value
            for j in range(0,len(truecontour_coords)):
                x_diff = abs(predcontour_coords[i][0] - truecontour_coords[j][0])
                y_diff = abs(predcontour_coords[i][1] - truecontour_coords[j][1])
                vector_diff = math.sqrt(x_diff**2 + y_diff**2)
                if vector_diff < shortest:
                    shortest = vector_diff
            distancesum += shortest
            pointschecked += 1
            
        #THIS SECTION ADDED TO MAKE IT A TWO-WAY COMPARISON METRIC    
        for i in range(0,len(truecontour_coords)):
            shortest = 1000      #initialize with arbitrarily large distance value
            for j in range(0,len(predcontour_coords)):
                x_diff = abs(truecontour_coords[i][0] - predcontour_coords[j][0])
                y_diff = abs(truecontour_coords[i][1] - predcontour_coords[j][1])
                vector_diff = math.sqrt(x_diff**2 + y_diff**2)
                if vector_diff < shortest:
                    shortest = vector_diff
            distancesum += shortest
            pointschecked += 1

        sliceaverage = distancesum / pointschecked
        total_surface_distance += sliceaverage
        sliceschecked += 1
    
    if sliceschecked > 0:
        mean_surface_distance = total_surface_distance / sliceschecked
    else:
        mean_surface_distance = 999.0
        
    return mean_surface_distance

def hausdorff_percent(truemask, predict, bilateral=False, percentile=95):    #finds the distance that 95% of points fall within
    distance_list = []
    
    for image in range(0,len(truemask)):
        
        if np.sum(truemask[image]) == 0:
            continue
        truecontours, heirarchy = cv2.findContours(truemask[image].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        truecontours = np.array(sorted(truecontours, key=len, reverse=True))
        truecontour_coords = truecontours[0]
        truecontour_coords = np.squeeze(truecontour_coords)
            #add clause for bilateral organs to repeat for the second largest single region
        if bilateral == True and len(truecontours) > 1:
            contour_coords = truecontours[1]
            contour_coords = np.squeeze(contour_coords)
            if len(contour_coords) > 2:
                truecontour_coords = np.concatenate((truecontour_coords,contour_coords), axis=0)

        if np.sum(predict[image]) == 0:
            continue
        predcontours, heirarchy = cv2.findContours(predict[image].astype('uint8'), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        predcontours = np.array(sorted(predcontours, key=len, reverse=True))
        predcontour_coords = predcontours[0]
        predcontour_coords = np.squeeze(predcontour_coords)
        if bilateral == True and len(predcontours) > 1:
            contour_coords = predcontours[1]
            contour_coords = np.squeeze(contour_coords)
            if len(contour_coords) > 2:
                predcontour_coords = np.concatenate((predcontour_coords,contour_coords), axis=0)
                
        for point in predcontour_coords:
            point = np.squeeze(point)
        for point in truecontour_coords:
            point = np.squeeze(point)
            
        if len(predcontour_coords.shape) < 2 or len(truecontour_coords.shape) < 2:
            continue
        
        for i in range(0,len(predcontour_coords)):
            shortest = 1000      #initialize with arbitrarily large distance value
            for j in range(0,len(truecontour_coords)):
                x_diff = abs(predcontour_coords[i][0] - truecontour_coords[j][0])
                y_diff = abs(predcontour_coords[i][1] - truecontour_coords[j][1])
                vector_diff = math.sqrt(x_diff**2 + y_diff**2)
                if vector_diff < shortest:
                    shortest = vector_diff
            distance_list.append(shortest) #inside the predmask point-by-point loop - each point has a value
            
            
        #SECTION ADDED TO MAKE IT A TWO-WAY COMPARISON METRIC
        for i in range(0,len(truecontour_coords)):
            shortest = 1000      #initialize with arbitrarily large distance value
            for j in range(0,len(predcontour_coords)):
                x_diff = abs(truecontour_coords[i][0] - predcontour_coords[j][0])
                y_diff = abs(truecontour_coords[i][1] - predcontour_coords[j][1])
                vector_diff = math.sqrt(x_diff**2 + y_diff**2)
                if vector_diff < shortest:
                    shortest = vector_diff
            distance_list.append(shortest) #inside the predmask point-by-point loop - each point has a value
        
        #loops through each slice, assigning each point of the prediction mask a hausdorff distance and appending that
            #distance to the list of distances
    
    if len(distance_list) > 0:
        #once all slices have been processed, simply sort list and find the 95th percentile
        distance_list.sort()
        cutoff = int(len(distance_list) * (percentile/100))
        percentile_value = distance_list[cutoff]
    else:
        percentile_value = 999.0
    
    return percentile_value
    
if __name__ == "__main__":
    resultspath = "G:\\machine learning misc\\outputs\\ParotidL"
    inputarray = np.load(resultspath + "\\test_data_input017_075.npy")    #will need to pull 
    maskarray = np.load(resultspath + "\\test_labels017_075.npy")
    CNNoutput = np.load(resultspath + "\\test_output017_075.npy")

    maskarray = np.squeeze(maskarray)
    CNNoutput = np.squeeze(CNNoutput)

    testprocess = output_postprocess(CNNoutput, bilateral = True)
    dsc = get_dicescore(maskarray,testprocess)
    TPF = sensitivity(maskarray, testprocess)
    TNF = specificity(maskarray, testprocess)
    msd = mean_surface_distance(maskarray, testprocess, bilateral = True)
    per95 = hausdorff_percent(maskarray, testprocess, bilateral = True, percentile=95)
    print("Dice Coefficient:",dsc)
    print("Sensitivty:", TPF)
    print("Specificity:",TNF)
    print("Mean Surfance Distance:",msd,"mm")
    print("95th Percentile Hausdorff Distance:", per95, "mm")