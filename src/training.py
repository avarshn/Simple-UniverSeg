import itertools
import pathlib
import subprocess
import random
from dataclasses import dataclass
from collections import defaultdict
from argparse import ArgumentParser, Namespace

import json
from time import time

import torchvision.transforms.v2 as transforms

import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import einops as E
import nibabel as nib
import PIL
import monai.losses

# from utils import const
# To ensure that modules within the project can be imported without specifying the full path
# sys.path is a list of directory paths where Python searches for modules when importing
# sys.path.append(PROJECT_ROOT_DIR) adds the project's root directory path to the sys.path. This effectively makes modules and packages within the project accessible from any location in the filesystem.
from pathlib import Path

import os
import sys
# Get the path of the parent directory of the current file (main.py)
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

# sys.path.append(Path("/projectnb/ec500kb/students/avarshn/Simple-UniverSeg/utils"))

from utils.const import PROJECT_ROOT_DIR, DATA_FOLDER, TENSORBOARD_LOG_DIR, MODEL_FOLDER
from utils.visualization import visualize_tensors, log_image
from utils.seed import seed_everything
from utils.dataset import create_data_loaders
from utils.dataset_with_varying_splits import create_data_loaders_v2
from models.original_universeg import universeg
from models.original_universeg.model import UniverSeg
from utils.metric import dice_score

from utils.add_argument import training_args
from utils.elastic_deformation import get_displacement

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


def test_UniverSeg(device: torch.device):
    model = universeg(pretrained=True).to(device)

    # Using Multi-GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    test_labels = [1, 8, 10, 11]
    support_set_sizes = [1, 2, 4, 8, 16, 32, 64]
    input_image, pred_image, gt, dice_scores = [], [], [], []
    for test_label in test_labels:
        d_support = example_data.OASISDataset("support", label=test_label)
        d_test = example_data.OASISDataset("test", label=test_label)

        support_images, support_labels = zip(*itertools.islice(d_support, 64))
        results = defaultdict(list)
        for support_set_size in support_set_sizes:
            support_images, support_labels = zip(
                *itertools.islice(d_support, support_set_size)
            )
            support_images = torch.stack(support_images).to(device)
            support_labels = torch.stack(support_labels).to(device)

            dices, hausdorffs = [], []
            for idx in range(len(d_test)):
                image, label = d_test[idx]
                image, label = image.to(device), label.to(device)
                # run inference
                logits = model(image[None], support_images[None], support_labels[None])[
                    0
                ].cpu()
                label = label.cpu()
                pred = torch.sigmoid(logits)
                hard_pred = pred.round().clip(0, 1)
                cur_dice = dice_score(hard_pred, label)

                input_image.append(image)
                pred_image.append(hard_pred)
                gt.append(label)
                dice_scores.append(cur_dice)

                # compute hausdorff distance
                hard_pred_one_hot = F.one_hot(hard_pred.long(), num_classes=2).permute(
                    0, 3, 1, 2
                )
                label_one_hot = F.one_hot(label.long(), num_classes=2).permute(
                    0, 3, 1, 2
                )

                cur_hausdorff = compute_hausdorff_distance(
                    hard_pred_one_hot, label_one_hot, percentile=95
                )

            print(
                f"Test label: {test_label}, support set size: {support_set_size}, dice: {np.mean(dices):.4f}, std:{np.std(dices):.4f}, hausdorff: {np.mean(hausdorffs):.4f}, std: {np.std(hausdorffs):.4f}"
            )


def sobel_edge_detection_conv2d(label_tensor):

    original_dtype = label_tensor.dtype
    label_tensor = label_tensor.float()

    # Define Sobel kernels for x and y directions
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
        dtype=label_tensor.dtype,
        device=label_tensor.device,
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
        dtype=label_tensor.dtype,
        device=label_tensor.device,
    ).view(1, 1, 3, 3)

    reshape_batches = False
    if len(label_tensor.shape) == 5:
        reshape_batches = True
        batch_size = label_tensor.shape[0]
        label_tensor = label_tensor.view(
            -1, 1, label_tensor.shape[3], label_tensor.shape[4]
        )

    # Apply convolutions to compute gradients
    gradient_x = F.conv2d(
        label_tensor, sobel_x, padding=1, groups=label_tensor.size(1)
    )  # Grouping along channels
    gradient_y = F.conv2d(
        label_tensor, sobel_y, padding=1, groups=label_tensor.size(1)
    )  # Grouping along channels

    # Compute gradient magnitude
    gradient_magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)

    # Round the gradient magnitude and clip values to ensure they are either 0 or 1
    binary_gradient = torch.clamp(torch.round(gradient_magnitude), 0, 1)

    # Convert binary gradient back to the original data type
    binary_gradient = binary_gradient.to(original_dtype)

    if reshape_batches:
        binary_gradient = binary_gradient.view(
            batch_size, -1, 1, binary_gradient.shape[2], binary_gradient.shape[3]
        )

    return binary_gradient


def task_augmentation(image, labels, support_images, support_labels):

    # Flip Intensities
    if np.random.rand() < 0.5:
        image = 1 - image
        support_images = 1 - support_images

    # Flip Labels
    if np.random.rand() < 0.5:
        labels = ~labels
        support_labels = ~support_labels

    # Flip horizontal
    if np.random.rand() < 0.5:
        image = transforms.functional.hflip(image)
        labels = transforms.functional.hflip(labels)
        support_images = transforms.functional.hflip(support_images)
        support_labels = transforms.functional.hflip(support_labels)

    # Flip vertical
    if np.random.rand() < 0.5:
        image = transforms.functional.vflip(image)
        labels = transforms.functional.vflip(labels)
        support_images = transforms.functional.vflip(support_images)
        support_labels = transforms.functional.vflip(support_labels)

    # Sobel-Edge Label
    if np.random.rand() < 0.5:
        labels = sobel_edge_detection_conv2d(labels)
        support_labels = sobel_edge_detection_conv2d(support_labels)

    # Affine Shift
    if np.random.rand() < 0.5:
        degree = random.randint(-15, 15)
        translate_x = random.uniform(0, 0.2)
        translate_y = random.uniform(0, 0.2)
        scale = random.uniform(0.8, 1.1)
        shear = 0
        image = transforms.functional.affine(
            image, degree, [translate_x, translate_y], scale, shear
        )
        labels = transforms.functional.affine(
            labels, degree, [translate_x, translate_y], scale, shear
        )
        support_images = transforms.functional.affine(
            support_images, degree, [translate_x, translate_y], scale, shear
        )
        support_labels = transforms.functional.affine(
            support_labels, degree, [translate_x, translate_y], scale, shear
        )

    # Elastic Warp
    if hyper_parameters["apply_elastic_warp_for_task_aug"]:
        print("Elastic Warp")
        if np.random.rand() < 0.1:
            alpha = random.uniform(1, 2.5)
            sigma = random.uniform(7, 8)
            _, height, width = transforms.functional.get_dimensions(image)
            displacement = get_displacement(alpha, sigma, [height, width])
            image = transforms.functional.elastic(image, displacement, transforms.InterpolationMode.BILINEAR)
            labels = transforms.functional.elastic(labels, displacement, transforms.InterpolationMode.NEAREST)
            support_images = transforms.functional.elastic(support_images, displacement, transforms.InterpolationMode.NEAREST)
            support_labels = transforms.functional.elastic(support_labels, displacement, transforms.InterpolationMode.NEAREST)

    # Brightness Contrast Change
    if np.random.rand() < 0.5:
        # Adjust brightness and contrast
        jitter = transforms.ColorJitter(brightness=0.1, contrast=0.5)
        image = jitter(image)
        support_images = jitter(support_images)

    # Sharpness Change
    if np.random.rand() < 0.5:
        sharpness_factor = random.uniform(0, 5)
        # Adjust sharpness
        image = transforms.functional.adjust_sharpness(image, sharpness_factor)
        support_images = transforms.functional.adjust_sharpness(
            support_images, sharpness_factor
        )

    # Gaussian Blur
    if np.random.rand() < 0.5:
        sigma_val = random.uniform(0.1, 1.1)
        image = transforms.functional.gaussian_blur(
            image, kernel_size=5, sigma=sigma_val
        )
        support_images = transforms.functional.gaussian_blur(
            support_images, kernel_size=5, sigma=sigma_val
        )

    # Gaussian Noise
    if np.random.rand() < 0.5:
        mean = random.uniform(0, 0.005)
        std = random.uniform(0, 0.06)  # std_square = [0, 0.0036]

        # noise = std * torch.randn_like(image) + mean
        # noise = std * torch.randn_like(support_images) + mean
        image = image + std * torch.randn_like(image) + mean
        support_images = support_images + std * torch.randn_like(support_images) + mean

    return image, labels, support_images, support_labels


def main(device: torch.device, writer: SummaryWriter, hyper_parameters: dict):
    model = UniverSeg(encoder_blocks=hyper_parameters["encoder_blocks"]).to(device)

    # Using Multi-GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    lr = hyper_parameters["lr"]
    iterations = hyper_parameters["iterations"]
    batch_size = hyper_parameters["batch_size"]
    support_set_size = hyper_parameters["support_set_size"]

    if hyper_parameters["Optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=hyper_parameters["weight_decay"]
        )
    elif hyper_parameters["Optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=hyper_parameters["momentum"],
            weight_decay=hyper_parameters["weight_decay"],
        )

    # Validation
    val_interval = hyper_parameters["val_interval"]
    num_workers = 4

    # Losses
    dice_loss = monai.losses.DiceLoss(to_onehot_y=True)
    dice_focal = monai.losses.DiceFocalLoss(
        to_onehot_y=True,
        lambda_dice = hyper_parameters["lambda_dice"],
        lambda_focal = hyper_parameters["lambda_focal"],
    )
    hausdorff_loss = monai.losses.HausdorffDTLoss(to_onehot_y=True)

    # Test Labels
    test_labels = {1, 8, 10, 11}
    train_labels = {label for label in range(1, 25)} - test_labels

    train_data_loaders, train_datasets = create_data_loaders_v2(
        train_labels,
        batch_size,
        num_workers,
        split="support",
        do_in_task_augmentation=hyper_parameters["do_in_task_augmentation"],
    )
    val_data_loaders, val_datasets = create_data_loaders_v2(
        test_labels, batch_size, num_workers, split="dev"
    )
    test_data_loaders, test_datasets = create_data_loaders_v2(
        test_labels, batch_size, num_workers, split="test"
    )

    step = 0
    best_dice = 0
    for iteration in range(iterations):
        print("Iteration", iteration + 1)
        model.train()

        task_idx = np.random.randint(20)
        cur_train_data_loader = train_data_loaders[task_idx]
        batch_data = next(iter(cur_train_data_loader))
        batch_indices = next(iter(cur_train_data_loader.batch_sampler))
        step += 1
        # [B, 1, H, W] - [Batch, grayscale - 1 channel, Height, Width]
        image, labels = batch_data[0].to(device), batch_data[1].to(device)

        support_images_batch, support_labels_batch = [], []
        for _ in range(batch_size):

            # Shuffled Support Set for every iteration
            support_set = [
                element
                for idx, element in enumerate(train_datasets[task_idx])
                if idx not in batch_indices
            ]
            # Shuffle the support_set list
            np.random.shuffle(support_set)

            # Subset based on support set size
            cur_support_images, cur_support_labels = zip(
                *itertools.islice(support_set, support_set_size)
            )
            cur_support_images = torch.stack(cur_support_images).to(device)
            cur_support_labels = torch.stack(cur_support_labels).to(device)
            support_images_batch.append(cur_support_images)
            support_labels_batch.append(cur_support_labels)

        # [B, Support Set Size, 1, H, W] - [Batch, Support Set Size, grayscale - 1 channel, Height, Width]
        support_images = torch.stack(support_images_batch)
        support_labels = torch.stack(support_labels_batch)

        # Task Augmentation
        if hyper_parameters["do_task_augmentation"]:
            image, labels, support_images, support_labels = task_augmentation(
                image, labels, support_images, support_labels
            )

        # Training
        optimizer.zero_grad()
        logits = model(image, support_images, support_labels)
        pred = torch.sigmoid(logits)
        one_hot_pred = torch.cat([1 - pred, pred], dim=1)

        if hyper_parameters["use_combined_loss"]:
            if iteration <= 2500:
                loss = dice_loss(
                    one_hot_pred,
                    labels,
                )
            if iteration > 2500:
                loss = dice_focal(
                    one_hot_pred,
                    labels,
                )
                loss += hyper_parameters["lambda_hausdorff"] * hausdorff_loss(
                    one_hot_pred,
                    labels,
                )

        else:
            # Only Dice loss
            loss = loss = dice_loss(
                    one_hot_pred,
                    labels,
                )

        # Backpropagation and Model Parameters Updatation
        loss.backward()
        optimizer.step()
        writer.add_scalar("train_loss/dice_loss", loss.item(), step)

        # Evalaution
        if iteration % val_interval == 0:
            model.eval()
            with torch.no_grad():
                (
                    dice_scores,
                    hausdorffs,
                    input_imgs,
                    label_imgs,
                    pred_img_softs,
                    pred_img_hards,
                ) = ([], [], [], [], [], [])
                for task_idx in range(4):
                    cur_dice_scores, cur_hausdorffs = [], []
                    cur_dev_data_loader = val_data_loaders[task_idx]
                    log_idx = 0
                    for dev_data, dev_indices in zip(
                        cur_dev_data_loader, cur_dev_data_loader.batch_sampler
                    ):
                        dev_image, dev_label = (
                            dev_data[0].to(device),
                            dev_data[1].to(device),
                        )
                        support_images_batch, support_labels_batch = [], []
                        cur_batch_size = dev_image.shape[0]
                        for _ in range(cur_batch_size):
                            support_set = [
                                element
                                for idx, element in enumerate(val_datasets[task_idx])
                                if idx not in dev_indices
                            ]
                            cur_support_images, cur_support_labels = zip(
                                *itertools.islice(support_set, support_set_size)
                            )
                            cur_support_images = torch.stack(cur_support_images).to(
                                device
                            )
                            cur_support_labels = torch.stack(cur_support_labels).to(
                                device
                            )

                            support_images_batch.append(cur_support_images)
                            support_labels_batch.append(cur_support_labels)

                        dev_support_images = torch.stack(support_images_batch)
                        dev_support_labels = torch.stack(support_labels_batch)
                        dev_logits = model(
                            dev_image, dev_support_images, dev_support_labels
                        )
                        dev_soft_pred = torch.sigmoid(dev_logits)
                        dev_hard_pred = dev_soft_pred.round().clip(0, 1)

                        if log_idx == 0:
                            input_imgs.append(
                                dev_image[iteration % dev_image.shape[0], 0, :, :]
                                .cpu()
                                .numpy()
                            )
                            label_imgs.append(
                                dev_label[iteration % dev_label.shape[0], 0, :, :]
                                .cpu()
                                .numpy()
                            )
                            pred_img_softs.append(
                                dev_soft_pred[
                                    iteration % dev_soft_pred.shape[0], 0, :, :
                                ]
                                .cpu()
                                .numpy()
                            )
                            pred_img_hards.append(
                                dev_hard_pred[
                                    iteration % dev_hard_pred.shape[0], 0, :, :
                                ]
                                .cpu()
                                .numpy()
                            )
                            log_idx += 1

                        cur_dice_scores.append(
                            torch.mean(dice_score(dev_hard_pred, dev_label)).item()
                        )
                        hard_pred_one_hot = (
                            F.one_hot(dev_hard_pred.long(), num_classes=2)
                            .permute(0, 4, 1, 2, 3)
                            .squeeze(2)
                        )
                        label_one_hot = (
                            F.one_hot(dev_label.long(), num_classes=2)
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
                    dice_scores.append(task_dice_scores)
                    hausdorffs.append(task_hausdorffs)
                    writer.add_scalar(
                        f"val_metric_task_{task_idx}/dice_score",
                        task_dice_scores,
                        iteration,
                    )
                    writer.add_scalar(
                        f"val_metric_task_{task_idx}/hausdorff_distance",
                        task_hausdorffs,
                        iteration,
                    )

                log_image(
                    writer,
                    input_imgs,
                    label_imgs,
                    pred_img_softs,
                    pred_img_hards,
                    iteration,
                )
                over_all_dice = sum(dice_scores) / len(dice_scores)
                over_all_hausdorff = sum(hausdorffs) / len(hausdorffs)
                writer.add_scalar(
                    "val_metric/overall_dice_score", over_all_dice, iteration
                )
                writer.add_scalar(
                    "val_metric/overall_hausdorff_distance",
                    over_all_hausdorff,
                    iteration,
                )

                if over_all_dice > best_dice:
                    best_dice = over_all_dice
                    print(
                        f"saving model at iteration {iteration + 1} with dice {best_dice:.4f} and hausdorff {over_all_hausdorff:.4f}"
                    )
                    if not (
                        MODEL_FOLDER / hyper_parameters["experiment_name"]
                    ).exists():
                        (MODEL_FOLDER / hyper_parameters["experiment_name"]).mkdir(
                            parents=True
                        )
                    torch.save(
                        model.state_dict(),
                        MODEL_FOLDER
                        / hyper_parameters["experiment_name"]
                        / f"model_{iteration + 1}_dice_{over_all_dice}_hausdorff_{over_all_hausdorff}.pth",
                    )


if __name__ == "__main__":
    start_time = time()
    print(f"Start time : {start_time}")
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser(description="Trainer args", add_help=False)
    training_args(parser)
    hparams = parser.parse_args()

    # Load hyperparameters from JSON file
    try:
        with open(hparams.config, "r") as file:
            hyper_parameters = json.load(file)
    except Exception as e:
        print(f"Failed to read or parse the JSON file: {str(e)}")
        exit(1)

    print(f"Hyperparameters: {hyper_parameters}")

    # To create Figure 3
    # pred_plot(device=device)

    # For table 1 and 2
    # test_UniverSeg(device=device)
    writer = SummaryWriter(
        log_dir=TENSORBOARD_LOG_DIR / hyper_parameters["experiment_name"]
    )

    # Add hyperparameters to TensorBoard
    for key, value in hyper_parameters.items():
        writer.add_text("Hyperparameters", f"{key}: {value}")

    main(device=device, writer=writer, hyper_parameters=hyper_parameters)
    end_time = time()

    print(
        f"Total time taken to run {hyper_parameters['experiment_name']} experiment - {end_time - start_time}"
    )
