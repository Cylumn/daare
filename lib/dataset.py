"""
Pytorch dataset for synthetic AKR.
Author: Allen Chang
Date Created: 08/02/2022
"""

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from data import simulate


class AKRDataset(Dataset):
    """
    Dataset class for loading and scaling synthetic AKR.
    """
    def __init__(self, n_dataset, args):
        """
        Initializes the dataset class.
        :param n_dataset: Size of the dataset.
        :param args: Command line arguments.
        """
        self.ground_truths = []
        self.observations = []
        brushes = simulate.read_brushes(args)

        for _ in range(n_dataset):
            y = simulate.ground_truth(brushes, args)
            x = simulate.noise(y, args)

            # Reshape and scale
            if args.disable_dataset_scaling:
                x = x[None, :, :]
                y = y[None, :, :]
            else:
                # Scale to N(0, 1)
                sc = StandardScaler()
                x = sc.fit_transform(x.reshape(-1, 1)).reshape(-1, args.img_size[0], args.img_size[1])
                y = sc.transform(y.reshape(-1, 1)).reshape(-1, args.img_size[0], args.img_size[1])
                # Rescale with mean and std
                x = x * args.dataset_intensity_scale[1] + args.dataset_intensity_scale[0]
                y = y * args.dataset_intensity_scale[1] + args.dataset_intensity_scale[0]

            # Add to dataset
            self.ground_truths.append(torch.tensor(y).float())
            self.observations.append(torch.tensor(x).float())

    def __len__(self):
        """
        :return: The length of the AKR dataset.
        """
        return len(self.ground_truths)

    def __getitem__(self, idx):
        """
        :return: A tuple of the (x, y) of the dataset.
        """
        return self.observations[idx], self.ground_truths[idx]