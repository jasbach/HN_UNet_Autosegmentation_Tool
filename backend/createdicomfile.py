# Coded version of DICOM file 'F:\testfolder\structureset.dcm'
# Produced by pydicom codify utility script
import os
import datetime
import random
import pydicom
import numpy as np
import cv2
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.sequence import Sequence

import evaluation_tool as eval_tool
import output_postprocess

"""
Will need to receive the image files to import certain metadata info
We'll need to loop through the image files and collect certain information
I don't think I want to build this into my other script that builds training arrays
because it's not a necessary element of building training data. It only is necessary
for this script which will be used as backend file generation to handle predictions
"""

def dataimport_dict(imagefile):
    
    patient_data = {}
    patient_data["PatientID"] = imagefile.PatientID
    patient_data["PatientName"] = imagefile.PatientName
    patient_data["PatientSex"] = imagefile.PatientSex
    patient_data["PatientBirthDate"] = imagefile.PatientBirthDate
    patient_data["PatientIdentityRemoved"] = imagefile.PatientIdentityRemoved
    patient_data["DeidentificationMethod"] = imagefile.DeidentificationMethod
    #patient_data["DeidentificationMethodCodeSequence"] = imagefile.DeidentificationMethodCodeSequence
    patient_data["StudyDate"] = imagefile.StudyDate
    patient_data["StudyTime"] = imagefile.StudyTime
    patient_data["AccessionNumber"] = imagefile.AccessionNumber
    patient_data["ReferringPhysicianName"] = imagefile.ReferringPhysicianName
    patient_data["StudyInstanceUID"] = imagefile.StudyInstanceUID
    patient_data["SeriesInstanceUID"] = imagefile.SeriesInstanceUID
    patient_data["FrameOfReferenceUID"] = imagefile.FrameOfReferenceUID
    patient_data["StudyID"] = imagefile.StudyID
    patient_data["SeriesNumber"] = imagefile.SeriesNumber
    patient_data["InstanceNumber"] = imagefile.InstanceNumber
    
    return patient_data

def gather_patient_data(rootfolder):
    
    listofimagepaths = []
    for root,dirs,files in os.walk(rootfolder):
        for name in files:
            if name.endswith(".dcm"):
                filepath = os.path.join(root,name)
                listofimagepaths.append(filepath)
    invalidfiles = 0
    UIDdict = {}
    datainitialized = False
    for filepath in listofimagepaths:
        imagefile = pydicom.read_file(filepath)
        if imagefile.Modality != "CT":
            print("File not a CT image, bypassing.")
            continue
        patient_data = dataimport_dict(imagefile)
        if datainitialized == False:
            check_dict = patient_data.copy()
            datainitialized = True
        try:
            check_dict["PatientID"] == patient_data["PatientID"]
        except:
            print("Mismatching patient ID found, bypassing",filepath)
            invalidfiles += 1
            continue
        sliceheight = round(imagefile.SliceLocation * 4) / 4 #rounds to nearest 0.25
        UIDdict[sliceheight] = (imagefile.SOPClassUID,imagefile.SOPInstanceUID)
    
    if invalidfiles > (len(listofimagepaths) * 0.1):
        raise Exception("Too many invalid files, double check input")
    
    return check_dict, UIDdict

def array_to_contour_coords(outputarray,heightlist,bilateral=False,image_size=256):
    contourelementlist = []
    for j in range(0,len(heightlist)):
        z_position = sorted(heightlist)[j]
        slice_to_add = outputarray[j].astype('uint8')
        if np.sum(slice_to_add) < 4:
            continue
        contours, heirarchy = cv2.findContours(slice_to_add, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = np.array(sorted(contours, key=len, reverse=True)) #orders contour sections from largest to smallest
        contour_coords = contours[0]  #chooses the largest single region tagged
        if len(contour_coords) < 4:
            continue
        contour_coords = contour_coords.flatten() - image_size/2  #reverts the origin shift to set 0,0 to patient center
        contour_coords = np.insert(contour_coords, range(2,len(contour_coords)+1,2), z_position)
        #inserts z-coord after every pair of coordinates
        contourelementlist.append(list(contour_coords))
        if bilateral == True and len(contours) > 1:
            contour_coords = contours[1] #chooses second largest single region tagged
            if len(contour_coords) < 4:
                continue
            contour_coords = contour_coords.flatten() - image_size/2
            contour_coords = np.insert(contour_coords, range(2,len(contour_coords)+1,2), z_position)
            contourelementlist.append(list(contour_coords))
    return contourelementlist

def generate_random_UID():
    UID = "0"
    while UID[0] == "0":
        UID = str(random.random())[2:] + str(random.random())[2:] + str(random.random())[-6:]
    return UID
        

def create_dicom(patient_data, UIDdict, structuresetdata,image_size=256,threshold=0.33):
    
    #structuresetdata will be a list of objects, each corresponding to an ROI
    #each list item will be a dictionary
    
    #Establish creation date and time for DICOM file
    currentyear = str(datetime.date.today().year)
    currentmonth = "%02d" % datetime.date.today().month
    currentday = "%02d" % datetime.date.today().day
    currentdate = currentyear + currentmonth + currentday
    
    currenthour = "%02d" % datetime.datetime.now().hour
    currentminute = "%02d" % datetime.datetime.now().minute
    currentsecond = "%02d" % datetime.datetime.now().second
    currenttime = currenthour + currentminute + currentsecond
    
    GeneratedInstanceUID = '1.2.246.352.221.' + generate_random_UID() #generates random 39-digit UID
    
    colordict = {"Brain":[0,0,198],"BrainStem":[251,216,151],"SpinalCord":[12,191,243],"ParotidL":[192,192,254],"ParotidR":[255,192,254],
                 "SubmandibularR":[255,255,187],"SubmandibularL":[200,249,134],"BrachialPlexus":[255,255,128],"Larynx":[192,241,254],
                 "CochleaL":[116,84,211],"CochleaR":[128,64,64]}
    
    ROInameconvert = {"BrachialPlexus":"Brachial Plexus","Brain":"Brain","CochleaL":"Cochlea L","CochleaR":"Cochlea R",
                  "Larynx":"Larynx","ParotidL":"Parotid L","ParotidR":"Parotid R","SpinalCord":"Spinal Cord","BrainStem":"Brainstem",
                  "SubmandibularL":"Submandibular L","SubmandibularR":"Submandibular R"}
    
    # File meta info data elements
    file_meta = FileMetaDataset()
    file_meta.FileMetaInformationGroupLength = 190
    file_meta.FileMetaInformationVersion = b'\x00\x01'
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    file_meta.MediaStorageSOPInstanceUID = GeneratedInstanceUID
    file_meta.TransferSyntaxUID = '1.2.840.10008.1.2'
    file_meta.ImplementationClassUID = '1.2.246.352.70.2.1.160.3'
    file_meta.ImplementationVersionName = 'DCIE 2.2'
    
    # Main data elements
    ds = Dataset()
    ds.SpecificCharacterSet = 'ISO_IR 192'
    ds.InstanceCreationDate = currentdate
    ds.InstanceCreationTime = currenttime
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3'
    ds.SOPInstanceUID = GeneratedInstanceUID
    ds.StudyDate = patient_data["StudyDate"] 
    ds.StudyTime = patient_data["StudyTime"]
    ds.AccessionNumber = patient_data["AccessionNumber"]
    ds.Modality = 'RTSTRUCT'
    ds.Manufacturer = 'NA'   #UNSURE
    ds.ReferringPhysicianName = patient_data["AccessionNumber"]
    
    # Coding Scheme Identification Sequence
    coding_scheme_identification_sequence = Sequence()
    ds.CodingSchemeIdentificationSequence = coding_scheme_identification_sequence
    
    # Coding Scheme Identification Sequence: Coding Scheme Identification 1
    coding_scheme_identification1 = Dataset()
    coding_scheme_identification1.CodingSchemeDesignator = 'FMA'                    #REVISIT THIS - WHAT TERMINOLOGY SET TO USE?
    coding_scheme_identification1.CodingSchemeUID = '2.16.840.1.113883.6.119'
    coding_scheme_identification_sequence.append(coding_scheme_identification1)
    
    
    # # Context Group Identification Sequence
    # context_group_identification_sequence = Sequence()
    # ds.ContextGroupIdentificationSequence = context_group_identification_sequence
    
    
    # # Context Group Identification Sequence: Context Group Identification 1
    # context_group_identification1 = Dataset()
    # context_group_identification1.MappingResource = '99VMS'
    # context_group_identification1.ContextGroupVersion = '20161209'
    # context_group_identification1.ContextIdentifier = 'VMS011'
    # context_group_identification1.ContextUID = '1.2.246.352.7.2.11'
    # context_group_identification_sequence.append(context_group_identification1)
    
    
    # # Mapping Resource Identification Sequence
    # mapping_resource_identification_sequence = Sequence()
    # ds.MappingResourceIdentificationSequence = mapping_resource_identification_sequence
    
    # # Mapping Resource Identification Sequence: Mapping Resource Identification 1
    # mapping_resource_identification1 = Dataset()
    # mapping_resource_identification1.MappingResource = '99VMS'
    # mapping_resource_identification1.MappingResourceUID = '1.2.246.352.7.1.1'
    # mapping_resource_identification1.MappingResourceName = 'Varian Medical Systems'
    # mapping_resource_identification_sequence.append(mapping_resource_identification1)
    
    ds.OperatorsName = ''
    ds.ManufacturerModelName = 'NA'   #UNSURE
    ds.PatientName =  patient_data["PatientName"]
    ds.PatientID = patient_data["PatientID"]
    ds.PatientBirthDate =  patient_data["PatientBirthDate"]
    ds.PatientSex = patient_data["PatientSex"]
    ds.PatientIdentityRemoved = patient_data["PatientIdentityRemoved"]
    ds.DeidentificationMethod = patient_data["DeidentificationMethod"]
    
    # De-identification Method Code Sequence
    #deidentification_method_code_sequence = Sequence()
    #ds.DeidentificationMethodCodeSequence = deidentification_method_code_sequence
    
    # De-identification Method Code Sequence: De-identification Method Code 1
    # deidentification_method_code1 = Dataset()              #WHOLE THING CAN BE IMPORTED?
    # deidentification_method_code1.CodeValue = '113100'
    # deidentification_method_code1.CodingSchemeDesignator = 'DCM'
    # deidentification_method_code1.CodeMeaning = 'Basic Application Confidentiality Profile'
    # deidentification_method_code_sequence.append(deidentification_method_code1)
    #deidentification_method_code_sequence.append(patient_data["DeidentificationMethodCodeSequence"])
    
    ds.SoftwareVersions = '4.2.7.0'
    ds.StudyInstanceUID = patient_data["StudyInstanceUID"]
    ds.SeriesInstanceUID = patient_data["SeriesInstanceUID"]   #NOT THE SAME AS SOPInstanceUID?? I believe need to just generate fresh number
    ds.StudyID = patient_data["StudyID"]
    ds.SeriesNumber = patient_data["SeriesNumber"]
    ds.InstanceNumber = patient_data["InstanceNumber"] + 1
    ds.LongitudinalTemporalInformationModified = 'REMOVED'
    ds.StructureSetLabel = 'DLC RTstruct'
    ds.StructureSetDate = currentdate
    ds.StructureSetTime = currenttime
    
    # Referenced Frame of Reference Sequence
    refd_frame_of_ref_sequence = Sequence()
    ds.ReferencedFrameOfReferenceSequence = refd_frame_of_ref_sequence
    
    # Referenced Frame of Reference Sequence: Referenced Frame of Reference 1
    refd_frame_of_ref1 = Dataset()
    refd_frame_of_ref1.FrameOfReferenceUID = patient_data["FrameOfReferenceUID"]
    
    # RT Referenced Study Sequence
    rt_refd_study_sequence = Sequence()
    refd_frame_of_ref1.RTReferencedStudySequence = rt_refd_study_sequence
    
    # RT Referenced Study Sequence: RT Referenced Study 1
    rt_refd_study1 = Dataset()
    rt_refd_study1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.481.3' #keep this the same
    rt_refd_study1.ReferencedSOPInstanceUID = patient_data["StudyInstanceUID"]
    
    # RT Referenced Series Sequence
    rt_refd_series_sequence = Sequence()
    rt_refd_study1.RTReferencedSeriesSequence = rt_refd_series_sequence
    
    # RT Referenced Series Sequence: RT Referenced Series 1
    rt_refd_series1 = Dataset()
    rt_refd_series1.SeriesInstanceUID = patient_data["SeriesInstanceUID"]
    
    # Contour Image Sequence
    contour_image_sequence = Sequence()
    rt_refd_series1.ContourImageSequence = contour_image_sequence
    
    for image in UIDdict: #need to figure out how to structure this
        contour_image = Dataset()
        contour_image.ReferencedSOPClassUID = UIDdict[image][0]
        contour_image.ReferencedSOPInstanceUID = UIDdict[image][1]
        contour_image_sequence.append(contour_image)
    
    rt_refd_series_sequence.append(rt_refd_series1)
    rt_refd_study_sequence.append(rt_refd_study1)
    refd_frame_of_ref_sequence.append(refd_frame_of_ref1)
    
    #----------Everything below this point relates to contours themselves--------------------------------------------------------
    
    # Structure Set ROI Sequence
    structure_set_roi_sequence = Sequence()
    ds.StructureSetROISequence = structure_set_roi_sequence
    
    
    for i in range(len(structuresetdata)):
        structure_set_roi = Dataset()
        structure_set_roi.ROINumber = i
        structure_set_roi.ReferencedFrameOfReferenceUID = patient_data["FrameOfReferenceUID"] #this is static, same for each element
        structure_set_roi.ROIName = ROInameconvert[structuresetdata[i][0]]
        structure_set_roi.ROIGenerationAlgorithm = 'AUTOMATIC'
        structure_set_roi_sequence.append(structure_set_roi)
    
    
    # ROI Contour Sequence
    roi_contour_sequence = Sequence()
    ds.ROIContourSequence = roi_contour_sequence
    
    height_limits = {"BrainStem":27,"ParotidL":33,"ParotidR":33,"SubmandibularL":17,
                     "SubmandibularR":17,"Larynx":18}
    #structuresetdata is list of lists, each element will be [name,3Darray,heightlist]
    for i in range(len(structuresetdata)):
        # ROI Contour Sequence: ROI Contour 1
        
        ROI = structuresetdata[i]
        
        ROIName = ROI[0]
        ROIarray = output_postprocess.apply_threshold(ROI[1],threshold)
        if ROIName in height_limits.keys():
            ROIarray = output_postprocess.height_prior(ROIarray,height_limits[ROIName])
        ROIarray = output_postprocess.scrap_stray(ROIarray)
        ROIarray = output_postprocess.simple_z_smoothing(ROIarray)
        ROIheightlist = ROI[2]
        
        
        roi_contour = Dataset()
        roi_contour.ROIDisplayColor = colordict[ROIName]#dictionary of approved colors by name [12, 191, 243]
        
        # Contour Sequence
        contour_sequence = Sequence()
        roi_contour.ContourSequence = contour_sequence
        
        if ROIName == "BrachialPlexus":
            bilateral = True
        else:
            bilateral = False
        listofelements = array_to_contour_coords(ROIarray,ROIheightlist, bilateral,image_size) #returns list of contour elements
        for contourelement in listofelements:
            sliceheight = contourelement[2] #each element is in x,y,z format, so element[2] is the z-coord
            contour = Dataset()
            #Contour Image Sequence
            contour_image_sequence = Sequence()
            contour.ContourImageSequence = contour_image_sequence
            #Contour Image Sequence: Contour Image 1
            contour_image1 = Dataset()
            contour_image1.ReferencedSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
            contour_image1.ReferencedSOPInstanceUID = UIDdict[sliceheight][1]
            contour_image_sequence.append(contour_image1)
            contour.ContourGeometricType = 'CLOSED_PLANAR'
            contour.NumberOfContourPoints = int(len(contourelement)/3)
            contour.ContourData = contourelement
            contour_sequence.append(contour)
        
        roi_contour.ReferencedROINumber = i
        roi_contour_sequence.append(roi_contour)
    
    # RT ROI Observations Sequence
    rtroi_observations_sequence = Sequence()
    ds.RTROIObservationsSequence = rtroi_observations_sequence
    
    for i in range(len(structuresetdata)):
        rtroi_observations = Dataset()
        rtroi_observations.ObservationNumber = i
        rtroi_observations.ReferencedROINumber = i
        rtroi_observations.ROIObservationLabel = structuresetdata[i][0]
        if "GTV" in structuresetdata[i][0]:
            rtroi_observations.RTROIInterpretedType = 'GTV' #'ORGAN' or 'PTV' or 'CTV' or 'GTV'
        elif "PTV" in structuresetdata[i][0]:
            rtroi_observations.RTROIInterpretedType = 'PTV'
        elif "CTV" in structuresetdata[i][0]:
            rtroi_observations.RTROIInterpretedType = 'CTV'
        else:
            rtroi_observations.RTROIInterpretedType = 'ORGAN'
        rtroi_observations.ROIInterpreter = ''
        rtroi_observations_sequence.append(rtroi_observations)
    
    
    ds.ApprovalStatus = 'UNAPPROVED'
    
    ds.file_meta = file_meta
    ds.is_implicit_VR = True
    ds.is_little_endian = True
    
    return ds

if __name__ == "__main__":
    testfolder = "F:\\DICOMdata\\RoswellData\\017_111"
    listofimagepaths = []
    heightlist = []
    for root,dirs,files in os.walk(testfolder):
        for name in files:
            if name.endswith(".dcm"):
                filepath = os.path.join(root,name)
                listofimagepaths.append(filepath)
                loadedfile = pydicom.read_file(filepath)
                if loadedfile.Modality == "CT":
                    heightlist.append(loadedfile.SliceLocation)
    
    ROIlist = ["BrachialPlexus","Brain","CochleaL","CochleaR","Larynx","ParotidL","ParotidR","SpinalCord",
                     "BrainStem","SubmandibularL","SubmandibularR"]
    ROInameconvert = {"BrachialPlexus":"Brachial Plexus","Brain":"Brain","CochleaL":"Cochlea L","CochleaR":"Cochlea R",
                      "Larynx":"Larynx","ParotidL":"Parotid L","ParotidR":"Parotid R","SpinalCord":"Spinal Cord","BrainStem":"Brainstem",
                      "SubmandibularL":"Submandibular L","SubmandibularR":"Submandibular R"}
    structuresetdata = []
    for ROI in ROIlist:
        print("Beginning work on %s" % ROI)
        if ROI == "BrachialPlexus":
            bilateral = True
        else:
            bilateral = False
        outputpath = "F:\\machine learning misc\\outputs\\local\\3D\\%s" % ROI  
        AxialOutput = np.load(outputpath + "\\Axial_prediction_017_111.npy")
        #CoronalOutput = np.load(outputpath + "\\Coronal_prediction_017_111.npy")
        #SagittalOutput = np.load(outputpath + "\\Sagittal_prediction_017_111.npy")
        MergedOutput = AxialOutput #*0.5 + CoronalOutput*0.25 + SagittalOutput*0.25               
        MergedOutput = np.squeeze(MergedOutput) 
        MergedOutput = eval_tool.output_postprocess(MergedOutput, bilateral)
        
        structuresetdata.append([ROInameconvert[ROI],MergedOutput,sorted(heightlist)])
                
    #structuresetdata is list of lists, each element will be [name,3Darray,heightlist]
    
    patient_data, UIDdict = gather_patient_data(testfolder)
    ds = create_dicom(patient_data,UIDdict,structuresetdata,image_size=256)
    ds.save_as(r'F:\testfolder\017_111structureset_from_codify.dcm', write_like_original=False)