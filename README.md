# Lacune-Detection

#### Method:
<img
src="images/Graphical_Abstract.png"
/>

# Automated detection of lacunes in brain MR images using SAM with robust prompts using self-distillation and anatomy-informed priors.
The method comprises of three different modules  
(i) **Candidate Prompt Generator (CPG)** - Segments lacunes and acts as prompt generator for SAM.  
(ii) **SAM Based Lacune Detection (SAM)** - Detects true lacunes based on multi-plane consistenty analysis.  
(iii) **Adaptive Region Thresholding (ART)** - Applies adaptive thresholds based on the regional preferences of lacunes. 

**Preprocessing:**  
We use FSL BET and FAST for brain extraction and bias field correction respectively.

**Training:**  
Only the CPG module requires training.  This module serves as a prompt generator and is trained using 2D axial slices of FLAIR images.
<pre>
For training the CPG module,  
Usage: python CPG_train.py  
Arguments:  
    --root_train        Path to folder containing preprocessed 2D axial slices of shape 256x256 for training. The script will process all files matching the pattern:*.nii.gz
    --root_trainmask    Path to folder containing preprocessed corresponding 2D masks of shape 256x256 for training.
    --root_val          Path to folder containing preprocessed 2D axial slices of shape 256x256 for validation.
    --root_valmask      Path to folder containing preprocessed corresponding 2D masks of shape 256x256 for validation.
    --folder_path       Path to folder for saving the best weights.
    --device            Device to be used to train the model.
    --batch_size        default set to 8.
    --learning_rate     default set to 1e-4.
    --epochs            No of epochs to train the model.
</pre>

**Inference:**   
<pre>
For preprocessing the test volumes for inference,  
Usage: bash PreprocessTest.sh   
Arguments:
    --input_dir    Path to the folder containing the Test 3D FLAIR NIfTI files.The script will process all files matching the pattern:*FLAIR.nii.gz  
    --output_dir    Path to folder for saving preprocessed Test Files
</pre>  
We have loaded the SAM ViT-H checkpoint to initialize the SAM model with pretrained weights obtained from:  
https://github.com/facebookresearch/segment-anything  

MARS region atlas in MNI space is taken from  
https://github.com/v-sundaresan/microbleed-size-counting 
<pre>
For inference on the preprocessed the test volumes,
Usage: bash Run_Inference.sh
Arguments:
    --device                    Device to be used to train the model.
    --sam_path                  Path to Segment Anything Model Code Directory.
    --sam_checkpoint            Path to SAM ViT-H checkpoint (.pth file).
    --root_test                 Path to folder containing the Preprocessed Test 3D FLAIR Volumes.The script will process all files matching the pattern:*preprocessed.nii.gz  
    --model_path                Path to trained weights of the CPG module. 
    --CPGSAM_output_path        Path to folder to save the CPG+SAM outputs.
    --ART_output_path           Path to folder to save the ART output.
    --Registered_MARS_files     Path to folder containing the MARS Atlas registered to Subject space.
</pre> 
