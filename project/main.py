import itertools
from collections import defaultdict
import pathlib
import subprocess
from dataclasses import dataclass
import random

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

import utils.dataset as example_data
from utils.visualization import visualize_tensors
from utils.utils import seed_everything, create_data_loaders
from models.original_universeg import universeg
from models.original_universeg.model import UniverSeg
from utils.metric import dice_score
from utils.const import DATA_FOLDER, TENSORBOARD_LOG_DIR


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

            # select an image, label test pair
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


def main(device: torch.device, writer: SummaryWriter):
    model = UniverSeg(encoder_blocks=[16, 16]).to(device)

    lr = 1e-3
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3
    )
    iterations = 414 * 20 * 100
    val_interval = 500
    support_set_size = 32
    batch_size = 16

    dice_loss = monai.losses.DiceLoss(to_onehot_y=True)
    test_labels = {1, 8, 10, 11}
    train_labels = {label for label in range(1, 25)} - test_labels

    train_data_loaders, train_datasets = create_data_loaders(
        train_labels, batch_size, support_set_size
    )
    val_data_loaders, val_datasets = create_data_loaders(
        test_labels, batch_size, support_set_size
    )

    step = 0
    for iteration in range(iterations):
        model.train()

        task_idx = np.random.randint(20)
        cur_train_data_loader = train_data_loaders[task_idx]
        batch_data = next(iter(cur_train_data_loader))
        batch_indices = cur_train_data_loader.batch_sampler
        step += 1
        image, labels = batch_data[0].to(device), batch_data[1].to(device)

        support_images_batch, support_labels_batch = [], []
        for _ in range(batch_size):
            support_set = [
                element
                for idx, element in enumerate(train_datasets[task_idx])
                if idx not in batch_indices
            ]
            cur_support_images, cur_support_labels = zip(
                *itertools.islice(support_set, support_set_size)
            )
            cur_support_images = torch.stack(cur_support_images).to(device)
            cur_support_labels = torch.stack(cur_support_labels).to(device)

            support_images_batch.append(cur_support_images)
            support_labels_batch.append(cur_support_labels)

        support_images = torch.stack(support_images_batch)
        support_labels = torch.stack(support_labels_batch)

        optimizer.zero_grad()
        logits = model(image, support_images, support_labels)
        pred = torch.sigmoid(logits)
        one_hot_pred = torch.cat([1 - pred, pred], dim=1)
        loss = dice_loss(one_hot_pred, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar("train_loss/dice_loss", loss.item(), step)
        print(f"current iteration: {iteration + 1}, current loss: {loss.item():.4f}")

        if (iteration + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for task_idx in range(4):
                    dice_scores, hausdorffs = [], []
                    cur_dev_data_loader = val_data_loaders[task_idx]
                    for dev_data, dev_indices in zip(
                        cur_dev_data_loader, cur_dev_data_loader.batch_sampler
                    ):
                        dev_image, dev_label = (
                            dev_data[0].to(device),
                            dev_data[1].to(device),
                        )
                        support_images_batch, support_labels_batch = [], []
                        for _ in range(batch_size):
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
                        dice_scores.append(dice_score(dev_hard_pred, dev_label))

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
                        hausdorffs.append(cur_hausdorff)

                dice_result = sum(dice_scores) / len(dice_scores)
                hausdorff_result = sum(hausdorffs) / len(hausdorffs)
                writer.add_scalar(
                    f"val_metric_task_{task_idx}/dice_score", dice_result, iteration
                )
                writer.add_scalar(
                    f"val_metric_{task_idx}/hausdorff_distance",
                    hausdorff_result,
                    iteration,
                )
                print(
                    f"current epoch: {iteration + 1}"
                    f" current dice score {task_idx}: {dice_result:.4f}"
                )

            torch.save(
                model.state_dict(),
                DATA_FOLDER
                / "UniverSeg_16_16"
                / f"model_{iteration + 1}_dice_{dice_result}_hausdorff_{hausdorff_result}.pth",
            )


if __name__ == "__main__":
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    # To create Figure 3
    # pred_plot(device=device)

    # For table 1 and 2
    # test_UniverSeg(device=device)
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR / "UniverSeg_32_32")
    main(device=device, writer=writer)
