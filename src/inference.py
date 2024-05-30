import itertools
import pathlib
import subprocess
from matplotlib import gridspec
import os
import random
from dataclasses import dataclass
from collections import defaultdict
from argparse import ArgumentParser, Namespace

import json
from time import time

import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import einops as E
from tqdm.auto import tqdm
import nibabel as nib
import PIL
import monai.losses
import pandas as pd
from torch.utils.data import DataLoader, Dataset


import utils.dataset as example_data
from utils.visualization import visualize_tensors, log_image
from utils.seed import seed_everything
from utils.dataset import create_data_loaders_noshuffle, create_data_loaders
from models.original_universeg import universeg
from models.original_universeg.model import UniverSeg
from utils.metric import dice_score
from utils.const import DATA_FOLDER, TENSORBOARD_LOG_DIR, MODEL_FOLDER
from utils.add_argument import add_argument




@torch.no_grad()
def inference(model, image, label, support_images, support_labels):
    image, label = image.to(device), label.to(device)

    # inference
    logits = model(image[None], support_images[None], support_labels[None])[
        0
    ]  # outputs are logits

    soft_pred = torch.sigmoid(logits)
    hard_pred = soft_pred.round().clip(0, 1)

    #  score
    score = dice_score(hard_pred, label)

    # return a dictionary of all relevant variables
    return {
        "Image": image,
        "Soft Prediction": soft_pred,
        "Prediction": hard_pred,
        "Ground Truth": label,
        "score": score,
    }




def test_UniverSeg(model_dir, result_dir, device: torch.device, ensemble=False, baseline=False):
    if (baseline):
        model = universeg(pretrained=True).to(device)
    else:
        model = UniverSeg(encoder_blocks=[8, 8]).to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir)))

    test_labels = [1, 8, 10, 11]
    # test_labels = 1
    batch_size = 4
    support_set_sizes = [1, 2, 4, 8, 16, 32, 64]

    if (ensemble):
        K = 10
    else:
        K = 1

    num_workers = 4
    test_data_loaders, test_datasets = create_data_loaders_noshuffle(
        test_labels, batch_size, num_workers, split="test"
    )
    support_data_loaders, support_datasets = create_data_loaders(
        test_labels, batch_size, num_workers, split="support"
    )

    task_dice_report = np.zeros((len(support_set_sizes), 12))
    task_hausdorffs_report = np.zeros((len(support_set_sizes), 12))
    total_hausdorffs = np.zeros((len(support_set_sizes), len(test_labels)))
    totall_dice = np.zeros((len(support_set_sizes), len(test_labels)))

    total_report = np.zeros((len(support_set_sizes), len(test_labels)))

    input_image = {'task1':[], 'task8':[], 'task10':[], 'task11':[]}
    hard_pred = {'task1':[], 'task8':[], 'task10':[], 'task11':[]}
    soft_pred = {'task1':[], 'task8':[], 'task10':[], 'task11':[]}
    gt = {'task1':[], 'task8':[], 'task10':[], 'task11':[]}



    for task_idx in range(4):
        cur_test_data_loader = test_data_loaders[task_idx]
        cur_support_data = support_datasets[task_idx]
        for si, support_set_size in enumerate(support_set_sizes):
            
            cur_dice_scores, cur_hausdorffs = [], [] 
            flag = 0           
            for test_data, test_indices in zip(
                cur_test_data_loader, cur_test_data_loader.batch_sampler
            ):
                test_image, test_label = (
                    test_data[0].to(device),
                    test_data[1].to(device),
                )            
                for k in range(K):
                    support_images_batch, support_labels_batch = [], []
                    cur_test_logits = []
                    for _ in range(batch_size):  
                        cur_support_images, cur_support_labels = zip(
                                    *itertools.islice(cur_support_data, support_set_size)
                                )
                        cur_support_images = torch.stack(cur_support_images).to(
                            device
                        )
                        cur_support_labels = torch.stack(cur_support_labels).to(
                            device
                        )
                        support_images_batch.append(cur_support_images)
                        support_labels_batch.append(cur_support_labels)

                    test_support_images = torch.stack(support_images_batch)
                    test_support_labels = torch.stack(support_labels_batch)
                    
                    cur_test_logits.append(model(
                        test_image, test_support_images, test_support_labels
                    ))    
                test_logits = sum(cur_test_logits)/len(cur_test_logits)
                test_soft_pred = torch.sigmoid(test_logits)
                test_hard_pred = test_soft_pred.round().clip(0, 1)
                if(flag == 0):
                    input_image[f'task{test_labels[task_idx]}'].append(test_image[0, 0, :, :])
                    hard_pred[f'task{test_labels[task_idx]}'].append(test_hard_pred[0, 0, :, :])
                    soft_pred[f'task{test_labels[task_idx]}'].append(test_soft_pred[0, 0, :, :])
                    gt[f'task{test_labels[task_idx]}'].append(test_label[0, 0, :, :])
                    flag = flag + 1

                #Get scores on current batch and current task
                cur_dice_scores.append(
                    torch.mean(dice_score(test_hard_pred, test_label)).item()
                )
                hard_pred_one_hot = (
                    F.one_hot(test_hard_pred.long(), num_classes=2)
                    .permute(0, 4, 1, 2, 3)
                    .squeeze(2)
                )
                label_one_hot = (
                    F.one_hot(test_label.long(), num_classes=2)
                    .permute(0, 4, 1, 2, 3)
                    .squeeze(2)
                )
                cur_hausdorff = compute_hausdorff_distance(
                    hard_pred_one_hot, label_one_hot, percentile=95
                )
                cur_hausdorff = cur_hausdorff.reshape(-1, 1)
                filtered_hausdorff = cur_hausdorff[
                    ~torch.any(cur_hausdorff.isnan(), dim=1)
                ]
                cur_hausdorffs.append(
                    torch.mean(filtered_hausdorff, dim=0).item()
                )

            task_dice_scores = sum(cur_dice_scores) / len(cur_dice_scores)
            task_hausdorffs = sum(cur_hausdorffs) / len(cur_hausdorffs)

            task_dice_report[si][3*(task_idx)+0] = task_dice_scores
            task_dice_report[si][3*(task_idx)+1] = np.mean(cur_dice_scores)
            task_dice_report[si][3*(task_idx)+2] = np.std(cur_dice_scores)

            task_hausdorffs_report[si][3*(task_idx)+0] = task_hausdorffs
            task_hausdorffs_report[si][3*(task_idx)+1] = np.mean(cur_hausdorffs)
            task_hausdorffs_report[si][3*(task_idx)+2] = np.std(cur_hausdorffs)

            print(
            f"Test label: {test_labels[task_idx]}, support set size: {support_set_size}, dice: {np.mean(cur_dice_scores):.4f}, std:{np.std(cur_dice_scores):.4f}, hausdorff: {np.mean(cur_hausdorffs):.4f}, std: {np.std(cur_hausdorffs):.4f}"
            )
            totall_dice[si][task_idx] = task_dice_scores
            total_hausdorffs[si][task_idx] = task_hausdorffs


    for i in range(len(support_set_sizes)):
        total_report[i][0] = np.mean(totall_dice[i][:])
        total_report[i][1] = np.std(totall_dice[i][:])
        total_report[i][2] = np.mean(total_hausdorffs[i][:])
        total_report[i][3]= np.std(total_hausdorffs[i][:])
    #Save everything
    dice_report_columns = ['label1 dice', 'label1 dice mean', 'label1 dice std', 
                    'label8 dice', 'label8 dice mean', 'label8 dice std', 
                    'label10 dice', 'label10 dice mean', 'label10 dice std', 
                    'label11 dice', 'label11 dice mean', 'label11 dice std']

    hausdorffs_report_columns = ['label1 hausdorffs', 'label1 hausdorffs mean', 'label1 hausdorffs std', 
    'label8 hausdorffs', 'label8 hausdorffs mean', 'label8 hausdorffs std', 
    'label10 hausdorffs', 'label10 hausdorffs mean', 'label10 hausdorffs std', 
    'label11 hausdorffs', 'label11 hausdorffs mean', 'label11 hausdorffs std']


    dicedf = pd.DataFrame(task_dice_report, columns = dice_report_columns)
    hausdorffsdf = pd.DataFrame(task_hausdorffs_report, columns = hausdorffs_report_columns)
    totall = pd.DataFrame(total_report, columns = ['dice_mean', 'dice_std', 'hausdorffs_mean', 'hausdorffs_std'])

    dicedf.to_csv(os.path.join(result_dir, 'task_dice_score.csv'), index=False)
    hausdorffsdf.to_csv(os.path.join(result_dir, 'task_hausdorffs.csv'), index=False)
    totall.to_csv(os.path.join(result_dir, 'totall.csv'), index=False)

    col_names = [
    "N = 1",
    "N = 2",
    "N = 4",
    "N = 8",
    "N = 16",
    "N = 32",
    "N = 64",
    ]
    titles_list = [
        "Input Image",
        "Label",
        "Soft Pred",
        "Hard Pred",
    ]
    nrows, ncols = 4, 7
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))

    for jdx in range(ncols):
        for idx in range(nrows):
            if idx == 0:
                axes[idx][jdx].imshow(input_image['task1'][jdx].detach().cpu().numpy(), cmap="gray")
            elif idx == 1:
                axes[idx][jdx].imshow(gt['task1'][jdx].detach().cpu().numpy(), cmap="gray")
            elif idx == 2:
                axes[idx][jdx].imshow(soft_pred['task1'][jdx].detach().cpu().numpy(), cmap="gray")
            elif idx == 3:
                axes[idx][jdx].imshow(hard_pred['task1'][jdx].detach().cpu().numpy(), cmap="gray")

            axes[idx][jdx].grid(False)
            axes[idx][jdx].set_xticks([])
            axes[idx][jdx].set_yticks([])
            if idx == 0:
                axes[idx][jdx].set_title(col_names[jdx])
            if jdx == 0:
                axes[idx][jdx].set_ylabel(titles_list[idx])

    # plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'task1.pdf'), dpi=300)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8))

    for jdx in range(ncols):
        for idx in range(nrows):     
            if idx == 0:
                axes[idx][jdx].imshow(input_image['task11'][jdx].detach().cpu().numpy(), cmap="gray")
            elif idx == 1:
                axes[idx][jdx].imshow(gt['task11'][jdx].detach().cpu().numpy(), cmap="gray")
            elif idx == 2:
                axes[idx][jdx].imshow(soft_pred['task11'][jdx].detach().cpu().numpy(), cmap="gray")
            elif idx == 3:
                axes[idx][jdx].imshow(hard_pred['task11'][jdx].detach().cpu().numpy(), cmap="gray")

            axes[idx][jdx].grid(False)
            axes[idx][jdx].set_xticks([])
            axes[idx][jdx].set_yticks([])
            if idx == 0:
                axes[idx][jdx].set_title(col_names[jdx])
            if jdx == 0:
                axes[idx][jdx].set_ylabel(titles_list[idx])

    # plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'task11.pdf'), dpi=300)





if __name__ == "__main__":
    start_time = time()
    print(f"Start time : {start_time}")
    seed_everything(42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    #model dir is path to saved model
    model_dir = ''
    #result dir is path to where you want to save your results
    result_dir = ''
    test_UniverSeg(model_dir, result_dir, device=device, ensemble=True, baseline=False)
    end_time = time()

    print(f"Total time taken to run test experiment - {end_time - start_time}")
