import numpy as np
import glob
import os
import nibabel as nib
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import scipy
from torch.utils.data import DataLoader
import torchvision
from natsort import natsorted
from augmentation import augment1
from utils.train_utils import lacune_dataset, determine_dice_metric, focal_tversky, VoxelwiseSupConLoss_inImage
from u2net_cl import U2NETP
import argparse
from torch.autograd import Variable
import torch.optim as optim
import datetime

def main(args):
    Train_Files = glob.glob(os.path.join(args.root_train, "*.nii.gz"), recursive=True)
    Train_MaskFiles = glob.glob(os.path.join(args.root_trainmask, "*.nii.gz"), recursive=True)
    Train_Files = natsorted(Train_Files)
    Train_MaskFiles = natsorted(Train_MaskFiles)

    Train_Files_Val = glob.glob(os.path.join(args.root_val, "*.nii.gz"), recursive=True)
    Train_MaskFiles_Val = glob.glob(os.path.join(args.root_valmask, "*.nii.gz"), recursive=True)
    Train_Files_Val = natsorted(Train_Files_Val)
    Train_MaskFiles_Val = natsorted(Train_MaskFiles_Val)


    train_dataset = lacune_dataset(Train_Files, Train_MaskFiles, transform=augment1)
    val_dataset = lacune_dataset(Train_Files_Val, Train_MaskFiles_Val, transform=None)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=2, drop_last=True)

    model = U2NETP(in_ch=1, out_ch=1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device=device)

    layer_name = 'stage1d.rebnconv1d.relu_s1'
    output_feature = None

    def hook(module, input, output):
        nonlocal output_feature
        output_feature = output.detach()

    layer = dict(model.named_modules())[layer_name]
    hook_handle = layer.register_forward_hook(hook)

    optimizer = optim.Adam(list(model.parameters()), lr=args.learning_rate)

    cl_criterion = VoxelwiseSupConLoss_inImage()
    cl_criterion = cl_criterion.to(device=device)


    os.makedirs(args.folder_path, exist_ok=True)
    loss_store = []
    best_dice = 0

    for epoch in range(1, args.epochs + 1):
        print('Epoch ', epoch, '/', args.epochs, flush=True)

        total_loss = 0
        model.train()
        batches = 0

        for batch_idx, train_dict in enumerate(train_loader):
            data = train_dict[0]
            label = train_dict[1]
            label = torch.round(label)
            label = label.to(device=device)
            data = Variable(data)
            label = Variable(label)
            data = data.to(device=device)

            if list(data.size())[0] == args.batch_size:
                batches += 1
                optimizer.zero_grad()
                seg_pred1 = model.forward(data)
                label = label[:, 0, :, :]
                label = torch.squeeze(label, axis=1)
                brain_mask = ((torch.squeeze(data, axis=1)) > 0.1) * 1
                preds = (seg_pred1 > 0.5) * 1
                if epoch <= 50:
                    cl_loss = cl_criterion(output_feature, label, brain_mask)
                else:
                    fp_mask = (torch.logical_and(preds[:, 0, :, :] == 1, label == 0)) * 1
                    cl_loss = cl_criterion(output_feature, label, brain_mask, subtracted_mask=fp_mask)
                label = torch.unsqueeze(label, axis=1)

                train_loss = focal_tversky(seg_pred1, label) + cl_loss
                train_loss.backward()
                optimizer.step()
                total_loss += train_loss
                if batch_idx % 400 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                        100. * (batch_idx + 1) / len(train_loader), train_loss.item()), flush=True)

        av_loss = (total_loss / batches).detach().cpu().numpy()
        print('Training set: Average loss: ', av_loss, flush=True)

        model.eval()
        batches = 0
        test_labels = []
        seg_results = []
        with torch.no_grad():
            for batch_idx, train_dict in enumerate(val_loader):
                data = train_dict[0]
                label = train_dict[1]
                data = torch.squeeze(data, dim=1)
                data = data.to(device=device)
                label = torch.round(label)
                label = label.to(device=device)
                if list(data.size())[0] == 8:
                    batches += 1
                    pred_mask = model.forward(data)
                    pred_mask = pred_mask.sigmoid().data.cpu().numpy().squeeze()
                    label = label.data.cpu().numpy().squeeze()
                    for j in range(0, label.shape[0]):
                        test_labels.append(label[j])
                        seg_results.append(pred_mask[j])

        test_labels = np.asarray(test_labels)
        seg_results = np.asarray(seg_results)
        val_loss = F.binary_cross_entropy(torch.from_numpy(seg_results), torch.from_numpy(test_labels))
        loss_store.append([train_loss, val_loss])
        val_dice = determine_dice_metric(torch.from_numpy(seg_results > 0.5), torch.from_numpy(test_labels))

        if val_dice > best_dice:
            best_dice = val_dice
            filename = f"best_epoch.pth"
            full_file_path = os.path.join(args.folder_path, filename)
            torch.save(model.state_dict(), full_file_path)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train segmentation model")
    parser.add_argument("--root_train", type=str, required=True, help="Root directory for training data")
    parser.add_argument("--root_trainmask", type=str, required=True, help="Root directory for training masks")
    parser.add_argument("--root_val", type=str, required=True, help="Root directory for validation data")
    parser.add_argument("--root_valmask", type=str, required=True, help="Root directory for validation masks")
    parser.add_argument("--folder_path", type=str, required=True, help="Folder path to save the best model")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to run the model on")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs for training")

    args = parser.parse_args()
    main(args)
