import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
from skimage.transform import resize
import torch.nn as nn 

class lacune_dataset(Dataset):
    def __init__(self, data_paths, mask_paths, transform=None):
        self.data_paths = data_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __getitem__(self, index):
        vol_dat = nib.load(self.data_paths[index]).get_fdata()
        mask_dat = nib.load(self.mask_paths[index]).get_fdata()
        if self.transform:
            vol_dat, mask_dat = self.transform(vol_dat, mask_dat)
        vol_dat = np.asarray(vol_dat)
        mean_int = np.mean(vol_dat)
        std_int = np.std(vol_dat)
        vol_dat = (vol_dat - mean_int) / (std_int + 1e-6)
        vol_dat = torch.from_numpy(vol_dat).float()
        mask_dat = torch.from_numpy(mask_dat)
        mask_dat = torch.unsqueeze(mask_dat, dim=0)
        vol_dat = torch.unsqueeze(vol_dat, dim=0)
        return vol_dat, mask_dat

    def __len__(self):
        return len(self.data_paths)


def determine_dice_metric(pred, target):
    smooth = 1.
    pred_vect = pred.contiguous().view(-1)
    target_vect = target.contiguous().view(-1)
    intersection = (pred_vect * target_vect).sum()
    dice = (2. * intersection + smooth) / (torch.sum(pred_vect) + torch.sum(target_vect) + smooth)
    return dice

def focal_tversky(preds, target, alpha=0.03, beta=0.97, epsilon=1e-6):
    focal = torchvision.ops.sigmoid_focal_loss(preds, target.float(), alpha=0.25, gamma=2, reduction='mean')
    preds = torch.sigmoid(preds)
    preds = preds.reshape(-1)
    target = target.reshape(-1)
    TP = (preds * target).sum()
    FP = ((1 - target) * preds).sum()
    FN = (target * (1 - preds)).sum()
    Tversky = (TP + epsilon) / (TP + alpha * FP + beta * FN + epsilon)
    Tversky = (1-Tversky)
    return Tversky + focal

class VoxelwiseSupConLoss_inImage(nn.Module):
    def __init__(self, temperature=0.07, device="cuda:1", num_voxels=10500):
        super().__init__()
        self.temperature = temperature
        self.struct_element = np.ones((5, 5, 5), dtype=bool)
        self.device = device
        self.max_pixels = num_voxels
        self.coefficient = 1

    def forward(self, Zs, pixel_mask, brain_mask, subtracted_mask=None):
        self.max_pixels = 10500
        self.device = "cuda:1"
        self.temperature = 0.07
        self.coefficient = 1
        number_of_features = Zs.shape[1]
        positive_mask = (pixel_mask == 1)
        if subtracted_mask is not None:
            negative_mask = (subtracted_mask == 1).squeeze(0, 1)
        elif brain_mask is not None:
            negative_mask = torch.logical_and(brain_mask == 1, pixel_mask == 0).squeeze(0)
        else:
            negative_mask = (pixel_mask == 0)
        Zs = Zs.permute(1, 0, 2, 3)
        positive_pixels = Zs[:, positive_mask].reshape(-1, number_of_features)
        negative_pixels = Zs[:, negative_mask].reshape(-1, number_of_features)

        if positive_pixels.shape[0] > self.max_pixels:
            random_indices = torch.arange(self.max_pixels)
            random_indices = torch.randperm(random_indices.size(0))
            positive_pixels = positive_pixels[random_indices]

        if positive_pixels.shape[0] < negative_pixels.shape[0]:
            random_indices = torch.arange(positive_pixels.shape[0])
            random_indices = torch.randperm(random_indices.size(0))
            negative_pixels = negative_pixels[random_indices]
        elif negative_pixels.shape[0] > self.max_pixels:
            random_indices = torch.arange(self.max_pixels)
            random_indices = torch.randperm(random_indices.size(0))
            negative_pixels = negative_pixels[random_indices]

        pixels = torch.cat([positive_pixels, negative_pixels])
        labels = torch.tensor([1] * positive_pixels.shape[0] + [0] * negative_pixels.shape[0]).to(self.device)
        pixels = F.normalize(pixels)
        dot = torch.matmul(pixels, pixels.T)
        dot = torch.div(dot, self.temperature)
        dot = F.normalize(dot)
        exp = torch.exp(dot)
        class_mask = torch.eq(labels, labels.unsqueeze(1))
        class_mask[torch.arange(len(labels)), torch.arange(len(labels))] = False
        positive_mask = exp * class_mask
        positive_mask[positive_mask == 0] = 1
        negative_mask = exp * (~class_mask)
        denominator = torch.sum(exp, dim=1) - torch.diagonal(exp)
        full_term = torch.log(positive_mask) - torch.log(denominator)
        loss = -(1 / len(labels)) * torch.sum(full_term) * self.coefficient
        return loss
    
