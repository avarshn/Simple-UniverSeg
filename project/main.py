import itertools

import torch
import numpy as np


import utils.dataset as example_data
from utils.visualization import visualize_tensors

from models.original_universeg import universeg


def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2 * (y_pred * y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()


def test_origin_UniverSeg(device: torch.device, test_labels: list[int]):
    model = universeg(pretrained=True).to(device)

    for test_label in test_labels:
        d_support = example_data.OASISDataset("support", label=1)
        d_test = example_data.OASISDataset("test", label=1)

        support_images, support_labels = zip(*itertools.islice(d_support, 16))
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
        for idx in range(len(d_test)):
            image, label = d_test[idx]
            image, label = image.to(device), label.to(device)

            # run inference
            logits = model(image[None], support_images[None], support_labels[None])[
                0
            ].to("cpu")
            pred = torch.sigmoid(logits)

            # visualize
            res = {"data": [image, label, pred, pred > 0.5]}
            titles = ["image", "label", "pred (soft)", "pred (hard)"]
            visualize_tensors(res, col_wrap=4, col_names=titles)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_labels = [1, 8, 10, 11]
    test_origin_UniverSeg(device=device, test_labels=test_labels)
