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

At this time, the tools for running new training iterations on the UNet structure are not available in the repository, but I do plan to add them soon once I've cleaned up the files a little bit.

Feel free to reach out to me at johnasba@buffalo.edu with any feedback.
