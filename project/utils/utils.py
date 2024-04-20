import os
import random
import numpy as np
import torch

import utils.dataset as example_data

from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_data_loaders(labels_idx, batch_size, num_workers):
    data_loaders = []
    datasets = []
    for task_idx in labels_idx:
        d_support = example_data.My_OASISDataset("support", task_idx)
        supportloader = DataLoader(
            d_support, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        data_loaders.append(supportloader)
        datasets.append(d_support)
    return data_loaders, datasets
