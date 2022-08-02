"""
Auroral Kilometric Radiation (AKR) simulation script.
Author: Allen Chang
Date Created: 08/02/2022
"""

import os
import imageio
import numpy as np
import pandas as pd


def read_brushes(args):
    """
    Reads the brushes from the data directory.
    :param args: Command line arguments
    :return: Pandas dataframe of brushes
    """
    if args.refresh_brushes_file:
        brushes = []
        # Read all files ending in '.jpg' in the data directory
        for file in os.listdir(f'{args.path_to_data}/brushes'):
            if file[-4:] == '.jpg':
                # Temporary buffer for brush reading
                # The first two indices are to store brush dimensions
                buffer_brush = np.ones(2 + args.img_size[0] * args.img_size[1])
                img = (imageio.v2.imread(f'{args.path_to_data}/brushes/{file}', as_gray=True)) / 255
                buffer_brush[:2] = img.shape
                # The remaining indices are to store brush values
                img = np.array(img).reshape(-1)
                buffer_brush[2:2 + len(img)] = img
                
                brushes.append(buffer_brush)

        # Convert to pandas dataframe for file io
        brushes = pd.DataFrame(brushes)
        # Invert brushes
        brushes.iloc[:, 2:] = 1 - brushes.iloc[:, 2:]
        # Trim unused space
        max_size = int((brushes[0] * brushes[1]).max() + 2)
        brushes = brushes.iloc[:, :max_size]
        # Save to file
        brushes.to_csv(f'{args.path_to_data}/brushes/brushes.csv', header=None, index=False)

        return brushes
    else:
        return pd.read_csv(f'{args.path_to_data}/brushes/brushes.csv', header=None)


def uniform(thetas: tuple,
            *dn: int):
    """
    Draws from the uniform distribution with np.random.rand().
    :param thetas: Bounds of the uniformly random variable.
    :param shape: Shape of the random variable.
    :return: The value of the uniformly random variable.
    """
    start, end = thetas
    return np.random.rand(*dn) * (end - start) + start


def gaussian(thetas: tuple,
             *dn: int):
    """
    Draws from the gaussian distribution with np.random.randn().
    :param thetas: The mean and std of the gaussian random variable.
    :param shape: Shape of the random variable.
    :return: The value of the gaussian random variable.
    """
    mean, std = thetas
    return np.random.randn(*dn) * std + mean


def ground_truth(brushes, args):
    """
    Simulates ground truth AKR.
    :param brushes: A pandas dataframe of brushes.
    :param args: Command line arguments.
    :return: A numpy array containing AKR.
    """
    img_size_arr = np.array(args.img_size)

    # Random Variables
    bg_intensity = uniform(args.theta_bg_intensity)
    n_akr = np.random.poisson(args.theta_n_akr) + 1
    pos = gaussian((img_size_arr / 2, img_size_arr / 3), n_akr, 2).astype(int)
    idx_brush = uniform((0, len(brushes)), n_akr).astype(int)
    akr_intensity = (np.abs(gaussian(args.theta_akr_intensity, n_akr)) + bg_intensity)
    h_flip = np.random.binomial(1, 0.5, n_akr)
    v_flip = np.random.binomial(1, 0.5, n_akr)

    # Generation
    bg = np.ones(args.img_size) * bg_intensity
    buffer_brushes = np.zeros(args.img_size)
    for i in range(n_akr):
        # Determine AKR position
        brush_shape = tuple(brushes.iloc[idx_brush[i], :2].astype(int))
        p = [pos[i, j] - int(brush_shape[j] / 2) for j in range(2)]
        p = tuple([slice(p[j], p[j] + brush_shape[j], 1) for j in range(2)])
        # Load and flip AKR
        w, h = buffer_brushes[p].shape
        brush = brushes.iloc[idx_brush[i], 2:2 + brush_shape[0] * brush_shape[1]].values.reshape(brush_shape)
        if h_flip[i] == 1:
            brush = brush[:, ::-1]
        if v_flip[i] == 1:
            brush = brush[::-1]
        # Draw AKR
        buffer_brushes[p] = np.maximum(buffer_brushes[p], akr_intensity[i] * brush[:w, :h])

    return np.array(np.maximum(bg, buffer_brushes))


def noise(x, args):
    """
    Adds noise to the ground truth AKR.
    :param x: A numpy array of ground truth AKR.
    :param args: Command line arguments.
    :return: A numpy array of observed AKR.
    """
    img = x.copy()

    # Loop to overlap channels
    for _ in range(2):
        # Random Variables
        gaussian_intensity = uniform(args.theta_gaussian_intensity)
        overall_channel_intensity = uniform(args.theta_overall_channel_intensity)
        n_channels = np.random.poisson(args.theta_n_channels) + 1
        channel_gap = np.random.exponential(args.img_size[0] / n_channels, n_channels).astype(int)
        channel_height = np.random.exponential(args.theta_channel_height, n_channels).astype(int) + 1
        channel_intensity = uniform(args.theta_channel_intensity, n_channels)

        # Generation
        mask = pd.DataFrame(np.zeros(args.img_size)).transpose()
        for i in range(n_channels):
            # Determine y position
            pos_y = int(np.sum(channel_gap[:i + 1]) - channel_height[i])
            # Gradient "fade-in" and "fade-out" near channel borders
            grad = np.concatenate([np.flip(1 - np.geomspace(0.1, 1, channel_height[i])),
                                   1 - np.geomspace(0.1, 1, channel_height[i])])
            # Draw channel
            w, h = mask.iloc[:, pos_y:int(pos_y + channel_height[i] * 2)].shape
            mask.iloc[:, pos_y:int(pos_y + channel_height[i] * 2)] = (grad * channel_intensity[i])[:h]

        mask = mask.transpose()

        # Draw noise
        img = img + overall_channel_intensity * np.abs(mask.values)
        img = img + gaussian_intensity * np.abs(gaussian((0, 1), args.img_size[0], args.img_size[1]))

    return img
