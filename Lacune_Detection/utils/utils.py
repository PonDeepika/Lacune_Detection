import numpy as np
import torch
from skimage import measure
import skimage
from skimage.transform import resize
from torch.utils.data import Dataset
import nibabel as nib
import os

def cut_zeros1d(im_array):
    im_list = list(im_array > 0)
    start_index = im_list.index(1)
    end_index = im_list[::-1].index(1)
    length = len(im_array[start_index:]) - end_index
    return start_index, end_index, length

def tight_crop_data(img_data, mask_data):
    row_sum = np.sum(np.sum(img_data, axis=1), axis=1)
    col_sum = np.sum(np.sum(img_data, axis=0), axis=1)
    stack_sum = np.sum(np.sum(img_data, axis=1), axis=0)
    rsid, reid, rlen = cut_zeros1d(row_sum)
    csid, ceid, clen = cut_zeros1d(col_sum)
    ssid, seid, slen = cut_zeros1d(stack_sum)
    return img_data[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen], mask_data[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen],  [rsid, rlen, csid, clen, ssid, slen]

def tight_crop_img(img_data):
    row_sum = np.sum(np.sum(img_data, axis=1), axis=1)
    col_sum = np.sum(np.sum(img_data, axis=0), axis=1)
    stack_sum = np.sum(np.sum(img_data, axis=1), axis=0)
    rsid, reid, rlen = cut_zeros1d(row_sum)
    csid, ceid, clen = cut_zeros1d(col_sum)
    ssid, seid, slen = cut_zeros1d(stack_sum)
    return img_data[rsid:rsid + rlen, csid:csid + clen, ssid:ssid + slen],  [rsid, rlen, csid, clen, ssid, slen]


def determine_dice_metric(pred, target):
    smooth = 1.
    pred_vect = pred.contiguous().view(-1)
    target_vect = target.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice

def get_bounding_boxes(mask, threshold=0.5):
    labeled_mask, numc = measure.label((mask > threshold) * 1, background=None, return_num=True, connectivity=2)
    bounding_boxes = [region.bbox for region in measure.regionprops(labeled_mask)]
    return bounding_boxes

def calculate_confusion_matrix(y_true, y_pred, threshold=0.5):
    true_boxes = get_bounding_boxes(y_true, threshold)
    pred_boxes = get_bounding_boxes(y_pred, threshold)
    confusion_matrix_arr = np.zeros((2, 2), dtype=int)
    for pred_box in pred_boxes:
        intersects = box_intersection(pred_box, y_true, threshold)
        min_x, min_y, min_z, max_x, max_y, max_z = pred_box
        crop_pred = y_pred[min_x:max_x, min_y:max_y, min_z:max_z]
        crop_pred = [crop_pred > threshold]
        crop_pred_bool = np.asarray(crop_pred).astype(bool)
        if not intersects:
            confusion_matrix_arr[0, 1] += 1
    for true_box in true_boxes:
        intersects = box_intersection(true_box, y_pred > threshold)
        if intersects:
            confusion_matrix_arr[1, 1] += 1
        else:
            confusion_matrix_arr[1, 0] += 1
    confusion_matrix_arr[0, 0] = 1 if not pred_boxes and not true_boxes else 0
    return confusion_matrix_arr

def box_intersection(bbox, mask, threshold=0.5):
    min_x, min_y, min_z, max_x, max_y, max_z = bbox
    cropped_volume = mask[min_x:max_x, min_y:max_y, min_z:max_z]
    return (np.sum(cropped_volume)) > 1

def helper_resize(image, output, result_nothresh, shape=(), crop_para=[], orig_shape=[]):
    shape = (int(shape[0]), int(shape[1]), int(shape[2]))
    image = skimage.transform.resize(image, output_shape=shape, order=1, preserve_range=True)
    output = skimage.transform.resize(output, output_shape=shape, order=0, preserve_range=True)
    result_nothresh = skimage.transform.resize(result_nothresh, output_shape=shape, order=0, preserve_range=True)
    actual_image = np.zeros(orig_shape)
    actual_output = np.zeros(orig_shape)
    act_result_unthresh = np.zeros(orig_shape)
    actual_image[crop_para[0]:crop_para[0] + crop_para[1], crop_para[2]:crop_para[2] + crop_para[3], crop_para[4]:crop_para[4] + crop_para[5]] = image
    actual_output[crop_para[0]:crop_para[0] + crop_para[1], crop_para[2]:crop_para[2] + crop_para[3], crop_para[4]:crop_para[4] + crop_para[5]] = output
    act_result_unthresh[crop_para[0]:crop_para[0] + crop_para[1], crop_para[2]:crop_para[2] + crop_para[3], crop_para[4]:crop_para[4] + crop_para[5]] = result_nothresh
    return actual_image, actual_output, act_result_unthresh


class axis3D_dataset(Dataset):  # Inherit from Dataset class
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, index):
        volume = nib.load(self.file_paths[index])
        filename = os.path.basename(self.file_paths[index])
        vol = volume.get_fdata()
        orig_shape = np.shape(vol)
        pix_dim =volume.header['pixdim']
        vol, crop_params = tight_crop_img(vol)
        cropped_shape = np.shape(vol)
        data=[]
        data_mmax=[]
        vol_sqrt = np.sqrt(vol)
        perc_99_val = np.percentile(vol_sqrt,99.5)
        vol_filt= vol_sqrt[vol_sqrt<perc_99_val]
        min_int = np.min(vol_filt)
        max_int = np.max(vol_filt)          
        for i in range(0, (np.shape(vol))[2]):
          slice=[]
          ax_slice = vol[:,:,i]
          ax_slice= resize(ax_slice,(256,256),order=1)
          ax_slice = np.asarray(ax_slice)
          ax_slice1 = ax_slice
          mean_int = np.mean(ax_slice)
          std_int = np.std(ax_slice)
          ax_slice= (ax_slice- mean_int) / (std_int+1e-6)        
          ax_slice1= np.sqrt(ax_slice1)
          ax_slice1= (ax_slice1- min_int) / (max_int - min_int+1e-6) 
          data_mmax.append(ax_slice1)
          slice.append(ax_slice)
          data.append(np.asarray(slice))      
        data=np.asarray(data)
        data_mmax= np.asarray(data_mmax)
        data = torch.from_numpy(data).float()  
        data_mmax = torch.from_numpy(data_mmax).float()           
        return data, data_mmax, cropped_shape, crop_params, pix_dim, orig_shape, filename

    def __len__(self):
        return len(self.file_paths)