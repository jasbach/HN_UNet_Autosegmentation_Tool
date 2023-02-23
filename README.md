# HN_UNet_Autosegmentation_Tool  
Accessible tool for using U-Net to run automatic segmentation on head and neck CT images  

This repo has three main subfolders: training, frontend, and backend  

"training" is a capture of the tools and scripts that can be used for both training or inference of the model (trained weights are provided in the repo).  
"backend" is the infrastructure to support serving the neural network via a JavaScript browser-based interface as a webserver  
"frontend" houses the JavaScript to support that browser-based interface.  

Note that this repo does not contain any files or structure for configuring the server itself. When I configured a prototype, I used Apache server software.  
Configuring that side of things is outside the scope of this project.  

If you are not interested in tinkering with the WebApp side of things and only want to use the neural network, you should be able to do that completely in the  
"training" folder. Note that you will need to run the scripts out of the folder that you download it to - this is not built to be installed as a package.  

# INSTRUCTIONS FOR INTERACTIVE USE

To use pre-trained weights to generate a new structure set of the 11 OARs that are discussed in the publication associated with this project, you want the file **generate_dicom.py**

Ensure you have the packages installed as listed in requirements.txt

Simply edit line 29 with the path to the folder containing your CT files, then run the script. It will generate a DICOM structure set file in the same folder as the script.

If you'd like to train your own neural network using the same process we used, refer to **training.py**  

Edit the paths in the first section of the file, then run the file. Note that the CT files for training data must be organized by patient in subfolders within the main folder that MAINROOTPATH points to.  
