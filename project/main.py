import itertools

import torch
import numpy as np
from monai.metrics import compute_hausdorff_distance
import torch.nn.functional as F

import utils.dataset as example_data
from utils.visualization import visualize_tensors
from utils.utils import seed_everything
from models.original_universeg import universeg


def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2 * (y_pred * y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()


def test_origin_UniverSeg(device: torch.device, test_labels: list[int]):
    model = universeg(pretrained=True).to(device)

    support_set_sizes = [1, 2, 4, 8, 16, 32, 64]
    for test_label in test_labels:
        d_support = example_data.OASISDataset("support", label=test_label)
        d_test = example_data.OASISDataset("test", label=test_label)

        support_images, support_labels = zip(*itertools.islice(d_support, 64))
        for support_set_size in support_set_sizes:
            support_images, support_labels = zip(
                *itertools.islice(d_support, support_set_size)
            )
            support_images = torch.stack(support_images).to(device)
            support_labels = torch.stack(support_labels).to(device)

            # n_viz = 10
            # visualize_tensors(
            #     {
            #         "Support Image": support_images[:n_viz],
            #         "Support Label": support_labels[:n_viz],
            #     },
            #     col_wrap=10,
            #     title="Support Set Examples",
            # )

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

                # compute hausdorff distance
                hard_pred_one_hot = F.one_hot(hard_pred.long(), num_classes=2).permute(
                    0, 3, 1, 2
                )
                label_one_hot = F.one_hot(label.long(), num_classes=2).permute(
                    0, 3, 1, 2
                )
                # hard_pred = torch.cat([1 - hard_pred, hard_pred], dim=0)
                # label = torch.cat([1 - label, label], dim=0)

                # hard_pred_one_hot_ = hard_pred_one_hot.unsqueeze(0)
                # label_one_hot_ = label_one_hot.unsqueeze(0)
                # print(
                #     f"hard_pred: {hard_pred_one_hot_.shape}, label: {label_one_hot_.shape}"
                # )

                print(f"hard_pred: {hard_pred}")
                cur_hausdorff = compute_hausdorff_distance(
                    hard_pred_one_hot, label_one_hot, percentile=95
                )
                dices.append(cur_dice)
                hausdorffs.append(cur_hausdorff.item())

                # visualize
                # res = {"data": [image, label, pred, pred > 0.5]}
                # titles = ["image", "label", "pred (soft)", "pred (hard)"]
                # visualize_tensors(res, col_wrap=4, col_names=titles)

            print(
                f"Test label: {test_label}, support set size: {support_set_size}, dice: {np.mean(dices):.4f}, std:{np.std(dices):.4f}, hausdorff: {np.mean(hausdorffs):.4f}, std: {np.std(hausdorffs):.4f}"
            )


if __name__ == "__main__":
    seed_everything(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_labels = [1, 8, 10, 11]
    test_origin_UniverSeg(device=device, test_labels=test_labels)
