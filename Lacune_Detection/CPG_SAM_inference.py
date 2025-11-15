import numpy as np
import glob
import os
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from natsort import natsorted
import skimage
import cv2
from segment_anything import sam_model_registry, SamPredictor
from u2net_cl import U2NETP
from utils.utils import cut_zeros1d, axis3D_dataset, determine_dice_metric, get_bounding_boxes, box_intersection, helper_resize
import argparse
from scipy.ndimage import binary_fill_holes
from skimage import measure
from skimage.measure import regionprops, label
from skimage.transform import resize
    

def main(args):
    # Initialize SAM (Segment Anything Model)
    model_type = "vit_h"
    device = args.device
    sam_checkpoint = args.sam_checkpoint
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
     # Load test data (FLAIR volumes and corresponding masks)
    root_test = args.root_test
    volume_test = os.path.join(root_test, "*preprocessed.nii.gz")  #Search Pattern
    Test_files= glob.glob(volume_test, recursive=True)
    Test_files= natsorted(Test_files)
    print("Number of TestFiles:",len(Test_files))
    test_dataset = axis3D_dataset(Test_files)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # Load the trained CPG segmentation model
    model = U2NETP(in_ch=1, out_ch=1)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device=device)
    #Set the CPG model to eval mode
    model.eval()

    
    threshold =0.5
    for i, pack in enumerate(test_dataloader, start=1):
        print("Processing Volume No:",i)     
        images, images_mmax, cropped_shape, crop_params,pix_dim, orig_shape, filename = pack
        if isinstance(filename, tuple):
               filename = filename[0]
        images = torch.squeeze(images,dim=1)
        images = torch.squeeze(images,dim=0) 
        images_mmax = torch.squeeze(images_mmax,dim=0)
        images_mmax= np.asarray(images_mmax)
        images_mmax[images_mmax<0.2]=0
        result=[]

        # Predictions of CPG on axial slices
        for j in range (0, (np.shape(images))[0]):
                img = images[j:j+1]
                img= img.to(device=device)
                model= model.to(device=device)
                res5= model(img)
                res5 = res5.sigmoid().data.cpu().numpy().squeeze()
                result.append(res5)
        images = torch.squeeze(images,dim=1)
        result = np.asarray(result)
        kernel = np.ones((3,3), np.uint8)
        result_unthresh = np.asarray(result).astype('float32')
        result = np.asarray(result>0.5).astype(np.uint8)
        result = cv2.dilate(result,kernel)        
        images = np.asarray(images)
        ori_vol_affine = nib.load(Test_files[i-1]).affine
        result_cpg = np.transpose(result, (1,2,0))
        act_images, act_output_cpg, act_output_unthresh_cpg =helper_resize(images, result_cpg, result_cpg, cropped_shape, crop_params, orig_shape)
        predict_nii=nib.Nifti1Image(act_output_cpg,affine=ori_vol_affine)
        filename_cpg = filename.replace("preprocessed.nii.gz", "PredictedCPG.nii.gz")
        full_file_path = os.path.join(args.CPGSAM_output_path, filename_cpg)
        nib.save(predict_nii, full_file_path)
        print("CPG Inference completed for Volume",i)


        bboxes = get_bounding_boxes((result>0.5)*1)
        sum_value = 350/pix_dim[0,1]
        conf_threshold =0.75
        if pix_dim[0,3]<4: # Inspect coronal and sagittal only is slice thickness is less than 4 mm.       
          for bbox in bboxes:
             #  Inspect the Coronal slice through the centroid using SAM
             Centroid= np.trunc([(bbox[0]+bbox[3])/2 , (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2]).astype(int)
             index = Centroid[2]
             slice = (images_mmax[:,:,index])
             slice=np.repeat(slice[..., np.newaxis], 3, axis=-1)
             predictor.set_image(slice)
             input_point = np.array([[Centroid[1], Centroid[0]]])
             input_label = np.array([1])
             masks, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label,multimask_output=True)
             first_mask = masks[0]
             first_score = scores[0]
            

             if(first_score>=conf_threshold):
                # Check for false positives from low contrast region / bigger structures
               if((np.sum(first_mask))>sum_value):
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]     
                        target_comp_mask =labelled_array== target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0
                        
               # Check for false positives from sulcus
               else:
                brain_mask = np.asarray((slice>0.2)*1).astype(np.uint8)              
                mask_data = brain_mask[:,:,0].astype(bool)
                filled_mask = binary_fill_holes(mask_data)
                filled_mask= np.asarray(filled_mask*1).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                filled_mask= cv2.erode(filled_mask,kernel)
                roi = filled_mask*first_mask
                sulcus = np.logical_xor(roi, first_mask)
                if((np.sum(sulcus))>1):
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array==target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0
                # Check for ellipticity and major/minor axis lengths
                else:
                  fmask_boxes = get_bounding_boxes((first_mask>0.5)*1)
                  for fbox in fmask_boxes:
                        min_x, min_y, max_x, max_y= fbox
                        crop_patch = np.asarray(slice[min_x:max_x, min_y:max_y]).astype(np.uint8)
                        crop_patch_mask = first_mask[min_x:max_x, min_y:max_y]
                        if (np.sum(crop_patch_mask)>1):
                            label_img = label(crop_patch_mask, connectivity=1)
                            props = regionprops(label_img)
                            minor_length = props[0].axis_minor_length
                            if (minor_length==0):
                                 minor_length =1
                            if ( props[0].axis_major_length/minor_length>=5  or props[0].axis_major_length<2):
                                 labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                                 target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                                 target_comp_mask =labelled_array==target_comp_label
                                 result[target_comp_mask]=0
                                 result_unthresh[target_comp_mask]=0

          result = np.asarray(result)
          bboxes = get_bounding_boxes((result>0.5)*1)
          for bbox in bboxes:
             #  Inspect the Sagittal slice through the centroid using SAM
             Centroid= np.trunc([(bbox[0]+bbox[3])/2 , (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2]).astype(int)
             index = Centroid[1]
             slice = (images_mmax[:,index,:])
             slice=np.repeat(slice[..., np.newaxis], 3, axis=-1)
             predictor.set_image(slice)
             input_point = np.array([[Centroid[2], Centroid[0]]])
             input_label = np.array([1])
             masks, scores, logits = predictor.predict(point_coords=input_point,point_labels=input_label, multimask_output=True)
             first_mask = masks[0]
             first_score = scores[0]
             if(first_score>=conf_threshold):
                # Check for false positives from low contrast region / bigger structures
               if((np.sum(first_mask))>sum_value):    
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array== target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0
                        
                # Check for false positives from sulcus
               else:
                brain_mask = np.asarray((slice>0.2)*1).astype(np.uint8)              
                mask_data = brain_mask[:,:,0].astype(bool)
                filled_mask = binary_fill_holes(mask_data)
                filled_mask= np.asarray(filled_mask*1).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                filled_mask= cv2.erode(filled_mask,kernel)
                roi = filled_mask*first_mask
                sulcus = np.logical_xor(roi, first_mask)
                if((np.sum(sulcus))>1):
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array==target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0

                # Check for ellipticity and major/minor axis lengths
                else:
                     fmask_boxes = get_bounding_boxes((first_mask>0.5)*1)
                     for fbox in fmask_boxes:
                        min_x, min_y, max_x, max_y= fbox
                        crop_patch = np.asarray(slice[min_x:max_x, min_y:max_y]).astype(np.uint8)
                        crop_patch_mask = first_mask[min_x:max_x, min_y:max_y]
                        if (np.sum(crop_patch_mask)>1):
                            label_img = label(crop_patch_mask, connectivity=1)
                            props = regionprops(label_img)
                            minor_length = props[0].axis_minor_length
                            if (minor_length==0):
                                 minor_length =1
                            if (props[0].axis_major_length/minor_length>=5  or props[0].axis_major_length<2):
                                 labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                                 target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                                 target_comp_mask =labelled_array==target_comp_label
                                 result[target_comp_mask]=0
                                 result_unthresh[target_comp_mask]=0
 


        result = np.asarray(result)
        bboxes = get_bounding_boxes((result>0.5)*1)
        for bbox in bboxes:
             #  Inspect the Axial slice through the centroid using SAM
             Centroid= np.trunc([(bbox[0]+bbox[3])/2 , (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2]).astype(int)
             index = Centroid[0]
             slice = (images_mmax[index,:,:])
             slice=np.repeat(slice[..., np.newaxis], 3, axis=-1)
             predictor.set_image(slice)
             input_point = np.array([[Centroid[2], Centroid[1]]])
             input_label = np.array([1])
             masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
             first_mask = masks[0]
             first_score = scores[0]
             if(first_score>=conf_threshold):
               # Check for false positives from low contrast region / bigger structures
               if((np.sum(first_mask))>sum_value): 
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array== target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0

                # Check for false positives from sulcus
               else:
                brain_mask = np.asarray((slice>0.2)*1).astype(np.uint8)              
                mask_data = brain_mask[:,:,0].astype(bool)
                filled_mask = binary_fill_holes(mask_data)
                filled_mask= np.asarray(filled_mask*1).astype(np.uint8)
                kernel = np.ones((3,3), np.uint8)
                filled_mask= cv2.erode(filled_mask,kernel)
                roi = filled_mask*first_mask   
                sulcus = np.logical_xor(roi, first_mask)
                if((np.sum(sulcus))>1):
                        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                        target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                        target_comp_mask =labelled_array==target_comp_label
                        result[target_comp_mask]=0
                        result_unthresh[target_comp_mask]=0

                # Check for ellipticity and major/minor axis lengths
                else:
                     fmask_boxes = get_bounding_boxes((first_mask>0.5)*1)
                     for fbox in fmask_boxes:
                        min_x, min_y, max_x, max_y= fbox
                        crop_patch = np.asarray(slice[min_x:max_x, min_y:max_y]).astype(np.uint8)
                        crop_patch_mask = first_mask[min_x:max_x, min_y:max_y]
                        if (np.sum(crop_patch_mask)>1):
                            label_img = label(crop_patch_mask, connectivity=1)
                            props = regionprops(label_img)
                            minor_length = props[0].axis_minor_length
                            if (minor_length==0):
                                 minor_length =1
                            if (props[0].axis_major_length>20 or props[0].axis_major_length/minor_length>=5 ):
                                 labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                                 target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                                 target_comp_mask =labelled_array==target_comp_label
                                 result[target_comp_mask]=0
                                 result_unthresh[target_comp_mask]=0
                                                               

        result = (result>0.5)*1
        result = np.asarray(result).astype('float32')
        images = np.asarray(images)
        result = np.transpose(result, (1,2,0))
        result_unthresh = np.transpose(result_unthresh, (1,2,0))
        images = np.transpose(images, (1,2,0))
        result_mask = np.asarray((result>0.5)*1)
        result_unthresh = result_unthresh*result_mask
        orig_shape = [int(p.item()) for p in orig_shape]
        act_images, act_output, act_output_unthresh =helper_resize(images, result, result_unthresh, cropped_shape, crop_params, orig_shape)

        # Remove any 3D candidate predictions that has maximum axial diameter less than 3mm
        result = (act_output_unthresh>0.5)*1
        result = np.asarray(result).astype('float32')
        kernel = np.ones((2,2), np.uint8)
     #    result = cv2.erode(result,kernel)
        pred_boxes = get_bounding_boxes(result>0.5)
        labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
        for pred_box in pred_boxes:
         unique_2 = False
         min_x, min_y, min_z, max_x, max_y, max_z = pred_box
         max_dia =0
         max_depth =0
         for n in range(min_z, max_z):
          crop_pred= result[min_x:max_x, min_y:max_y, np.round(n).astype(int)]
          crop_pred = [crop_pred>threshold]
          crop_pred_bool = np.asarray(crop_pred).astype(bool)
          if (np.unique(crop_pred_bool).shape[0] >1):
            unique_2 = True
            labeled_mask = label(crop_pred_bool)
            props = regionprops(labeled_mask[0])
            new_dia = (props[0].axis_major_length)/pix_dim[0,1]
            if new_dia>max_dia:
                max_dia = new_dia
                max_depth =n

         if max_dia <= 3 and unique_2 == True:
                  labelled_array,numc = measure.label((result>0.5)*1, return_num=True,connectivity=3)
                  Centroid= np.trunc([(min_x + max_x)/2 , (min_y + max_y)/2, (min_z + max_z)/2]).astype(int)
                  if (result[Centroid[0],Centroid[1], Centroid[2]]==0 and (np.sum(result[min_x:max_x,min_y:max_y,max_depth]))>1):
                      Centroid[0] =np.where(result[min_x:max_x,min_y:max_y,max_depth]==1)[0][0]+min_x
                      Centroid[1] =np.where(result[min_x:max_x,min_y:max_y,max_depth]==1)[1][0]+min_y
                      Centroid[2] = max_depth
                  target_comp_label = labelled_array[Centroid[0],Centroid[1],Centroid[2]]
                  target_comp_mask =labelled_array==target_comp_label
                  result[target_comp_mask]=0
                  act_output_unthresh[target_comp_mask]=0
        
     #     Remove any 3D candidate that touches the outer brain region as sulcus
        result = np.asarray(result).astype('float32')
        result_mask = np.asarray((result>0.5)*1)
        result_unthresh = act_output_unthresh*result_mask
        act_output = np.array(result_unthresh>0).astype(np.uint8)    
        result_mask = np.asarray(result_mask).astype('float32')

        img = nib.load(Test_files[i-1]).get_fdata()
        min_int = np.min(img)
        max_int = np.max(img)
        act_images = img
        act_images= (act_images- min_int) / (max_int - min_int+1e-6) 
        brain_mask = np.asarray((act_images>0.15)*1).astype(np.uint8)              
        mask_data = brain_mask.astype(bool)
        filled_mask = binary_fill_holes(mask_data)
        filled_mask= np.asarray(filled_mask*1).astype(np.uint8)
        labels = label(act_output>0.5)
        unique_labels = np.unique(labels)[1:]  
        new_act_output = np.zeros_like(act_output)
        for label1 in unique_labels:
          mask = (labels == label1)
          roi = filled_mask*mask
          sulcus = np.logical_xor(roi, mask)
          if((np.sum(sulcus))>1):
           new_act_output[mask] = 0
           result_unthresh[mask]=0
          else:
           new_act_output[mask] = result_unthresh[mask]  
        ori_vol_affine = nib.load(Test_files[i-1]).affine
        predict_nii=nib.Nifti1Image(result_unthresh,affine=ori_vol_affine)
        filename = filename.replace("preprocessed.nii.gz", "PredictedCPGSAM.nii.gz")
        full_file_path = os.path.join(args.CPGSAM_output_path, filename)
        nib.save(predict_nii, full_file_path)
        print("CPG+SAM Inference completed for Volume",i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run segmentation model on test data")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on")
    parser.add_argument("--sam_checkpoint", type=str, required=True, help="Path to SAM checkpoint")
    parser.add_argument("--root_test", type=str, required=True, help="Root directory for testing data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--CPGSAM_output_path", type=str, required=True, help="Path to folder for saving CPG+SAM output")
    args = parser.parse_args()
    main(args)
