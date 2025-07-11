import numpy as np
import skimage
import nibabel as nib
import cv2
from utils.utils import get_bounding_boxes
from skimage import measure
import argparse
import os
import glob
from natsort import natsorted


def main(args):

  SAM_files = os.path.join(args.CPGSAM_output_path[0],  "*.nii.gz")  #Search Pattern
  SAM_unthresh_files= glob.glob(SAM_files, recursive=True)
  SAM_unthresh_files= natsorted(SAM_unthresh_files)

  MARS_files = os.path.join(args.Registered_MARS_files[0],  "*.nii.gz")  #Search Pattern
  Test_Atlasfiles= glob.glob(MARS_files, recursive=True)
  Test_Atlasfiles= natsorted(Test_Atlasfiles)

  possible_values = np.unique(np.round(SAM_unthresh_files[0].get_fdata()))
  possible_values = possible_values[1:]
  # Threshold for white matter, frontal, Internal capsule and External capsule = 0.5
  # Threshold for Basal Ganglia, Temporal, Cerebellum, Parietal, Insular = 0.55
  # Threshold for Thalamus, Hippocampus, Brain stem, Occipital = 0.65

  thresholds = [0.65, 1, 0.55, 0.65, 0.5, 0.5, 0.65, 0.5,0.5,0.55,0.55, 0.65, 0.55]
  thresholds = [threshold+0.1  for threshold in thresholds] #threhold incremented by 0.1 for VALDO
  for i in range(0,len(SAM_unthresh_files)):
    mask = nib.load(SAM_unthresh_files[i]).get_fdata()
    mask[:,:,int(0.8*mask.shape[2]):] =0
    mask_unthresh = mask
    mask = (mask>=0.5)*1
    kernel = np.ones((3,3), np.uint8)
    mask = np.asarray(mask).astype(np.uint8) 
    mask1 =mask
    kernel = np.ones((1,1), np.uint8)
    mask1 = cv2.erode(mask,kernel)
    pred_boxes = get_bounding_boxes(mask)
    atlas_values = nib.load(Test_Atlasfiles[i]).get_fdata()
    if (len(pred_boxes))>0:
        for bbox in pred_boxes:
            min_x, min_y,min_z, max_x, max_y, max_z= bbox
            centroid_x, centroid_y, centroid_z = np.round((min_x+max_x)/2).astype(int), np.round((min_y+max_y)/2).astype(int), np.round((min_z+max_z)/2).astype(int)
            centroid_mask_val = mask_unthresh[np.round(centroid_x).astype(int), np.round(centroid_y).astype(int), np.round(centroid_z).astype(int)]
            region_value = atlas_values[np.round((min_x+max_x)/2).astype(int), np.round((min_y+max_y)/2).astype(int), np.round((min_z+max_z)/2).astype(int)]
            index = np.where(possible_values == np.round(region_value))[0][0]
            if (centroid_mask_val <= thresholds[index]):
                labelled_array,numc = measure.label((mask>0)*1, return_num=True,connectivity=3)
                target_comp_label = labelled_array[np.round(centroid_x).astype(int), np.round(centroid_y).astype(int), np.round(centroid_z).astype(int)]
                target_comp_mask =labelled_array==target_comp_label
                mask[target_comp_mask]=0
    mask = np.asarray((mask>0)*1).astype('float')
    ori_vol_affine = nib.load(SAM_unthresh_files[i]).affine
    predict_nii=nib.Nifti1Image(mask,affine=ori_vol_affine)
    filename = f"predicted_ART{i}.nii.gz"
    full_file_path = os.path.join(args.ART_output_path[0], filename)
    nib.save(predict_nii, full_file_path)
    print("ART Inference completed for Volume",i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation model on test data")
    parser.add_argument("--CPGSAM_output_path", type=str, nargs='+',required=True, help="Path to folder containing list of unthresholded SAM output files ")
    parser.add_argument("--Registered_MARS_files", type=str,nargs='+', required=True, help="List of all MARS maps registerd to test files")
    parser.add_argument("--ART_output_path", type=str, nargs='+', required=True, help="Path to folder for saving ART output")
    parser.add_argument("--root_test", type=str, required=True, help="Root directory for testing data")
    args = parser.parse_args()
    main(args)
