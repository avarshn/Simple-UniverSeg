import math
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import einops as E
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import utils.dataset as example_data
from models.original_universeg import universeg
from utils.metric import dice_score


def visualize_tensors(tensors, col_wrap=1, col_names=None, title=None):
    M = len(tensors)
    N = len(next(iter(tensors.values())))

    cols = col_wrap
    rows = math.ceil(N / cols) * M

    d = 2.5
    fig, axes = plt.subplots(rows, cols, figsize=(d * cols, d * rows))
    if rows == 1:
        axes = axes.reshape(1, cols)

    for g, (grp, tensors) in enumerate(tensors.items()):
        for k, tensor in enumerate(tensors):
            col = k % cols
            row = g + M * (k // cols)
            x = tensor.detach().cpu().numpy().squeeze()
            ax = axes[row, col]
            if len(x.shape) == 2:
                ax.imshow(x, vmin=0, vmax=1, cmap="gray")
            else:
                ax.imshow(E.rearrange(x, "C H W -> H W C"))
            if col == 0:
                ax.set_ylabel(grp, fontsize=16)
            if col_names is not None and row == 0:
                ax.set_title(col_names[col])

    for i in range(rows):
        for j in range(cols):
            ax = axes[i, j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])

    if title:
        plt.suptitle(title, fontsize=20)

    return plt.gcf()


def pred_plot(device: torch.device):
    model = universeg(pretrained=True).to(device)

    test_labels = [1]
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
                break

        # visualize
        # using matplotlib to draw 7 col and 3 row image with title for each image on the first row
        d = 2.5
        rows, cols = 3, 7
        fig, axes = plt.subplots(rows, cols, figsize=(d * cols, d * rows))
        col_names = [
            "N = 1",
            "N = 2",
            "N = 4",
            "N = 8",
            "N = 16",
            "N = 32",
            "N = 64",
        ]
        if rows == 1:
            axes = axes.reshape(1, cols)

        grp = ["Input Image", "Prediction", "Ground Truth"]
        for g, (grp, tensors) in enumerate(zip(grp, [input_image, pred_image, gt])):
            for k, tensor in enumerate(tensors):
                col = k % cols
                row = g + 7 * (k // cols)
                x = tensor.detach().cpu().numpy().squeeze()
                ax = axes[row, col]
                if len(x.shape) == 2:
                    ax.imshow(x, vmin=0, vmax=1, cmap="gray")
                else:
                    ax.imshow(E.rearrange(x, "C H W -> H W C"))
                if col == 0:
                    ax.set_ylabel(grp, fontsize=16)
                if col_names is not None and row == 0:
                    ax.set_title(col_names[col])

        for i in range(rows):
            for j in range(cols):
                ax = axes[i, j]
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.savefig("/projectnb/ec500kb/projects/UniverSeg/code/plot.pdf", dpi=300)


def log_image(
    writer: SummaryWriter,
    input_imgs: torch.Tensor,
    label_imgs: torch.Tensor,
    pred_img_softs: torch.Tensor,
    pred_img_hards: torch.Tensor,
    step: int,
):
    titles_list = [
        "Input Image",
        "Label",
        "Pred (Soft)",
        "Pred (Hard)",
    ]
    fig = plt.figure(figsize=(20, 20))
    nrows, ncols = 4, 4
    gs = gridspec.GridSpec(nrows=nrows, ncols=ncols)
    for idx in range(nrows):
        for jdx in range(ncols):
            ax = plt.subplot(gs[idx * ncols + jdx])
            if jdx == 0:
                ax.imshow(np.rot90(input_imgs[idx], 2), cmap="gray")
            elif jdx == 1:
                ax.imshow(np.rot90(label_imgs[idx], 2), cmap="gray")
            elif jdx == 2:
                ax.imshow(np.rot90(pred_img_softs[idx], 2), cmap="gray")
            elif jdx == 3:
                ax.imshow(np.rot90(pred_img_hards[idx], 2), cmap="gray")
            ax.grid(False)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.set_xticks([])
            ax.set_yticks([])
            if idx == 0:
                ax.set_title(titles_list[idx * ncols + jdx])
            if jdx == 0:
                ax.set_ylabel(f"Task {idx}")

    plt.tight_layout()
    writer.add_figure(f"Iteration: {step} Val Image", fig, global_step=step)
    return None
