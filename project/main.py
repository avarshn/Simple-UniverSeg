import itertools
import pathlib
import subprocess
import random
from dataclasses import dataclass
from collections import defaultdict
from argparse import ArgumentParser, Namespace

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
from utils.visualization import visualize_tensors, log_image
from utils.utils import seed_everything, create_data_loaders
from models.original_universeg import universeg
from models.original_universeg.model import UniverSeg
from utils.metric import dice_score
from utils.const import DATA_FOLDER, TENSORBOARD_LOG_DIR, RESULT_FOLDER
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
    model = UniverSeg(encoder_blocks=[16, 8]).to(device)

    lr = 1e-3
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-3
    )
    iterations = 414 * 20 * 100
    val_interval = 100
    support_set_size = 32
    batch_size = 16
    num_workers = 4

    dice_loss = monai.losses.DiceLoss(to_onehot_y=True)
    test_labels = {1, 8, 10, 11}
    train_labels = {label for label in range(1, 25)} - test_labels

    train_data_loaders, train_datasets = create_data_loaders(
        train_labels, batch_size, num_workers, split="support"
    )
    val_data_loaders, val_datasets = create_data_loaders(
        test_labels, batch_size, num_workers, split="dev"
    )
    test_data_loaders, test_datasets = create_data_loaders(
        test_labels, batch_size, num_workers, split="test"
    )

    step = 0
    best_dice = 0
    for iteration in range(iterations):
        model.train()

        task_idx = np.random.randint(20)
        cur_train_data_loader = train_data_loaders[task_idx]
        batch_data = next(iter(cur_train_data_loader))
        batch_indices = next(iter(cur_train_data_loader.batch_sampler))
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
                    if not (RESULT_FOLDER / hparams.experiment_name).exists():
                        (RESULT_FOLDER / hparams.experiment_name).mkdir(parents=True)
                    torch.save(
                        model.state_dict(),
                        RESULT_FOLDER
                        / hparams.experiment_name
                        / f"model_{iteration + 1}_dice_{over_all_dice}_hausdorff_{over_all_hausdorff}.pth",
                    )


if __name__ == "__main__":
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = ArgumentParser(description="Trainer args", add_help=False)
    add_argument(parser)
    hparams = parser.parse_args()
    # device = "cpu"

    # To create Figure 3
    # pred_plot(device=device)

    # For table 1 and 2
    # test_UniverSeg(device=device)
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR / hparams.experiment_name)
    main(device=device, writer=writer)
