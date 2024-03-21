import itertools
from collections import defaultdict

import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance
import torch.nn.functional as F
import matplotlib.pyplot as plt
import einops as E

import utils.dataset as example_data
from utils.visualization import visualize_tensors
from utils.utils import seed_everything
from models.original_universeg import universeg


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


def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2 * (y_pred * y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()


def pred_plot(device: torch.device):
    model = universeg(pretrained=True).to(device)

    # test_labels = [1, 8, 10, 11]
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

                # # compute hausdorff distance
                # hard_pred_one_hot = F.one_hot(hard_pred.long(), num_classes=2).permute(
                #     0, 3, 1, 2
                # )
                # label_one_hot = F.one_hot(label.long(), num_classes=2).permute(
                #     0, 3, 1, 2
                # )

                # # hard_pred = torch.cat([1 - hard_pred, hard_pred], dim=0)
                # # label = torch.cat([1 - label, label], dim=0)

                # # hard_pred_one_hot_ = hard_pred_one_hot.unsqueeze(0)
                # # label_one_hot_ = label_one_hot.unsqueeze(0)
                # # print(
                # #     f"hard_pred: {hard_pred_one_hot_.shape}, label: {label_one_hot_.shape}"
                # # )

                # cur_hausdorff = compute_hausdorff_distance(
                #     hard_pred_one_hot, label_one_hot, percentile=95
                # )
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

        # if not np.isnan(cur_dice):
        #     dices.append(cur_dice)
        # else:
        #     print(
        #         "Dice NA test_label",
        #         test_label,
        #         "support_set_size",
        #         support_set_size,
        #         "idx",
        #         idx,
        #     )

        # if not np.isnan(cur_hausdorff.item()):
        #     hausdorffs.append(cur_hausdorff.item())
        # else:
        #     print(
        #         "HD95 test_label",
        #         test_label,
        #         "support_set_size",
        #         support_set_size,
        #         "idx",
        #         idx,
        #     )

        # visualize
        # res = {"data": [image, label, pred, pred > 0.5]}
        # titles = ["image", "label", "pred (soft)", "pred (hard)"]
        # visualize_tensors(res, col_wrap=4, col_names=titles)

        # print(
        #     f"Test label: {test_label}, support set size: {support_set_size}, dice: {np.mean(dices):.4f}, std:{np.std(dices):.4f}, hausdorff: {np.mean(hausdorffs):.4f}, std: {np.std(hausdorffs):.4f}"
        # )


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

                # # compute hausdorff distance
                # hard_pred_one_hot = F.one_hot(hard_pred.long(), num_classes=2).permute(
                #     0, 3, 1, 2
                # )
                # label_one_hot = F.one_hot(label.long(), num_classes=2).permute(
                #     0, 3, 1, 2
                # )

                # # hard_pred = torch.cat([1 - hard_pred, hard_pred], dim=0)
                # # label = torch.cat([1 - label, label], dim=0)

                # # hard_pred_one_hot_ = hard_pred_one_hot.unsqueeze(0)
                # # label_one_hot_ = label_one_hot.unsqueeze(0)
                # # print(
                # #     f"hard_pred: {hard_pred_one_hot_.shape}, label: {label_one_hot_.shape}"
                # # )

                # cur_hausdorff = compute_hausdorff_distance(
                #     hard_pred_one_hot, label_one_hot, percentile=95
                # )
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

        # if not np.isnan(cur_dice):
        #     dices.append(cur_dice)
        # else:
        #     print(
        #         "Dice NA test_label",
        #         test_label,
        #         "support_set_size",
        #         support_set_size,
        #         "idx",
        #         idx,
        #     )

        # if not np.isnan(cur_hausdorff.item()):
        #     hausdorffs.append(cur_hausdorff.item())
        # else:
        #     print(
        #         "HD95 test_label",
        #         test_label,
        #         "support_set_size",
        #         support_set_size,
        #         "idx",
        #         idx,
        #     )

        # visualize
        # res = {"data": [image, label, pred, pred > 0.5]}
        # titles = ["image", "label", "pred (soft)", "pred (hard)"]
        # visualize_tensors(res, col_wrap=4, col_names=titles)

        # print(
        #     f"Test label: {test_label}, support set size: {support_set_size}, dice: {np.mean(dices):.4f}, std:{np.std(dices):.4f}, hausdorff: {np.mean(hausdorffs):.4f}, std: {np.std(hausdorffs):.4f}"
        # )


if __name__ == "__main__":
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # To create Figure 3
    pred_plot(device=device)

    # For table 1 and 2
    test_UniverSeg(device=device)
