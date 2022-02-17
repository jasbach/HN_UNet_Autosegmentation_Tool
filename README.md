## NOTE: This application is under active development. Work in progress branches can be found in the repository. "main" branch will contain the most recent full release of the tool. The branch "BMC_RadOnc" will be preserved as the code was at the time of publication: https://ro-journal.biomedcentral.com/articles/10.1186/s13014-022-01982-y 

# HN_UNet_Autosegmentation_Tool
 Accessible tool for using U-Net to run automatic segmentation on head and neck CT images for clinical use.
 
 
Main file is **generate_dicom.py** - this file can be run in a Python interpreter or by command-line. All other files/folders support this process and should be stored in the same working directory as generate_dicom.py.

The input for the process is a string of the path to a folder. This folder should contain the DICOM image files for a single CT study for a head and neck patient. The burden is on the user to ensure that only one study's files exist in the folder or in subdirectories. The script can only handle one study at a time, and should be run separately for each patient you wish to contour.

Note that only the following organs are segmented by this tool:
- Brain
- Brainstem
- Cochlea L
- Cochlea R
- Parotid L
- Parotid R
- Submandibular L
- Submandibular R
- Brachial Plexus
- Larynx
- Spinal Cord

A script for running your own training iterations on the same UNet architecture is also provided: **training.py**

This script cannot be run command-line as packaged, you'll need to open it in an interpreter and modify the settings as needed. Both training and predicting depend on the support scripts for data handling - please download the entire repository into one location to ensure proper functionality.

CURRENT LIMITATIONS OF TRAINING TOOLS:
- Configured only for 256x256 images with pixel size 1mm by 1mm. Skeleton is there to support other configurations, I need to finalize the code for it first.
- Only can be used for CT images
- Requires good alignment of images (consistent slice thickness, contour data aligning with SliceLocation, etc). I've tried to build flags and errors in as many places as I could think of to catch issues if feed data needs cleaning. Future releases will be made more robust in this sense.
- Limited to CT arrangement where coordinates origin is in the center of each image. Can't map contours correctly otherwise. This will be updated in a future release.

Feel free to reach out to me at johnasba@buffalo.edu with any feedback.
