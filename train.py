'''
Training script for DAARE
Author: Allen Chang
Date Created: 08/02/2022
'''

import torch
import argparse
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data import simulate
from lib.dataset import AKRDataset
from model.daare import DAARE


def init_dataset(args):
    """
    Initializes the training and validation datasets.
    :param args: Command line arguments.
    :return: Returns a tuple (loader_train, loader_valid) of the training dataloader and the validation dataloader.
    """
    # Training dataset
    if args.verbose:
        print(f'> Loading training dataset of size {args.n_train}')
    data_train = AKRDataset(args.n_train, args)
    # Validation dataset
    if args.verbose:
        print(f'> Loading validation dataset of size {args.n_valid}')
    data_valid = AKRDataset(args.n_valid, args)
    loader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    loader_valid = DataLoader(data_valid, batch_size=args.batch_size, shuffle=True)

    return loader_train, loader_valid


def init_model(args):
    """
    Initializes the DAARE model, devices, and torch environment parameters.
    :param args: Command line arguments.
    :return: Returns a tuple (daare, device) of the DataParallel container for DAARE and the available device.
    """
    # GPU Speedup
    torch.backends.cudnn.benchmark = True
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    daare = nn.DataParallel(DAARE(depth=args.depth,
                                  hidden_channels=args.n_hidden,
                                  kernel=args.kernel,
                                  norm=True,
                                  img_size=args.img_size),
                            device_ids=args.device_ids)
    daare.to(device)
    # Log model
    if args.verbose:
        print(f'> Device: {device}')
        print(f'> DAARE Model:')
        print(f'\tDepth: {args.depth}')
        print(f'\tHidden Channels: {args.n_hidden}')
        print(f'\tKernel: {args.kernel}')

    return daare, device


def get_loss(criterion: nn.Module,
             daare: nn.Module,
             x: torch.Tensor,
             y: torch.Tensor):
    """
    Calculates loss between true noise and predicted noise.
    :param criterion: The criterion to use to calculate difference.
    :param daare: The DAARE model.
    :param x: The input AKR observation.
    :param y: The ground truth AKR.
    :return: A Tensor containing a single loss with grad.
    """
    # Calculate intermediate observation and noise predictions
    x_inter, z_inter = daare(x, return_intermediate=True)
    noise = x_inter - y
    return criterion(z_inter, noise)


def train_daare(criterion: nn.Module,
                daare: nn.Module,
                opt: torch.optim.Optimizer,
                x: torch.Tensor,
                y: torch.Tensor):
    """
    Back-propagates DAARE with the given optimizer.
    :param criterion: The criterion to use to calculate difference.
    :param daare: The DAARE model.
    :param opt: The optimizer.
    :param x: The input AKR observation.
    :param y: The ground truth AKR.
    :return: A float containing the loss value.
    """
    # Zero the gradients
    opt.zero_grad()
    # Calculate loss and update
    loss = get_loss(criterion, daare, x, y)
    loss.backward()
    opt.step()
    return loss.item()


def run_epoch(n_loader: int,
              loader: DataLoader,
              is_train: bool,
              daare: nn.Module,
              criterion: nn.Module,
              opt: torch.optim.Optimizer,
              device: torch.device,
              writer: SummaryWriter,
              idx_component: int,
              idx_epoch: int,
              args):
    """
    Runs a single epoch across a given dataloader.
    :param n_loader: The number of samples in the dataloader
    :param loader: The dataloader.
    :param is_train: Whether this epoch should be run in train or validation mode.
    :param daare: The DAARE model.
    :param criterion: The loss criterion.
    :param opt: The optimizer.
    :param device: The device to train on.
    :param writer: The logs writer.
    :param idx_component: The index of the current component.
    :param idx_epoch: The index of the current epoch.
    :param args: Command line arguments.
    :return: Total loss from the epoch.
    """
    # Set DAARE to the right mode
    if is_train:
        daare.train()
    else:
        daare.eval()

    # Run epoch
    loss_total = 0
    n_batches = int(n_loader / loader.batch_size)
    for idx_batch, data in tqdm(enumerate(loader), total=n_batches,
                                position=0, leave=True, bar_format=args.tqdm_format):
        # Load data
        x, y = data[0].to(device), data[1].to(device)
        # Back-propagate on DAARE
        if is_train:
            loss = train_daare(criterion, daare, opt, x, y) / n_batches
        else:
            loss = get_loss(criterion, daare, x, y).item() / n_batches

        # Log loss
        if not args.disable_logs:
            writer.add_scalar(f'Component {idx_component} loss/{("train" if is_train else "valid")}',
                              loss,
                              (idx_epoch - 1) * n_batches + idx_batch)
        loss_total += loss

    return loss_total


def start_training(args):
    # Initialization
    loader_train, loader_valid = init_dataset(args)
    daare, device = init_model(args)
    mse_loss = nn.MSELoss()

    # Set DAARE to train
    daare.train()

    # Logs
    if args.verbose:
        print(f'> Use logs: {not args.disable_logs}')
    if not args.disable_logs:
        writer = SummaryWriter(f'{args.path_to_logs}/{args.model_name}')
    else:
        writer = None

    if args.verbose:
        print(f'> Begin training for {args.n_cdae} components')
    # Loop for each CDAE component
    for idx_component in range(args.n_cdae):
        # Add a new CDAE component
        daare.module.add_cdae(residual=(idx_component > 0), norm=(idx_component < args.n_norm))
        # Init optimizer
        opt = torch.optim.Adam(daare.parameters(), lr=args.learning_rate)

        # Training Loop
        for idx_epoch in range(1, args.n_epochs_per_cdae + 1):
            print(f"CDAE[{idx_component}]: Epoch {idx_epoch} of {args.n_epochs_per_cdae}")

            # Train
            loss_train = run_epoch(n_loader=args.n_train, loader=loader_train, is_train=True,
                      daare=daare, criterion=mse_loss, opt=opt, device=device,
                      writer=writer, idx_component=idx_component, idx_epoch=idx_epoch, args=args)
            # Validation
            loss_valid = run_epoch(n_loader=args.n_valid, loader=loader_valid, is_train=False,
                      daare=daare, criterion=mse_loss, opt=opt, device=device,
                      writer=writer, idx_component=idx_component, idx_epoch=idx_epoch, args=args)

            # Flush logs
            if not args.disable_logs:
                writer.flush()

            # Print
            print(f"loss_train: {loss_train * 1e4:7.2f}", end=' | ')
            print(f"loss_valid: {loss_valid * 1e4:7.2f}")

        # Close logs
        writer.close()

    # Save model
    state_dict = {
        'state_dict': daare.state_dict(),
        'args': args
    }
    torch.save(state_dict, f'{args.out_path}/{args.model_name}.pt')


def get_args():
    parser = argparse.ArgumentParser('DAARE', add_help=False)

    # Paths
    parser.add_argument('--path_to_data', default='data', type=str, help='Path to the data directory.')
    parser.add_argument('--path_to_logs', default='logs', type=str, help='Path to the logs directory.')
    parser.add_argument('--out_path', default='./', type=str, help='Path to the output directory.')

    # Hardware
    parser.add_argument('--device_ids', default=[0, 1], type=int, nargs=2,
                        help="Device ids of the GPUs, if GPUs are available.")

    # Options
    parser.add_argument('--model_name', default='daare_v1', type=str, help='Name of the model when logging and saving.')
    parser.add_argument('--verbose', action='store_true', help='Trains with debugging outputs and print statements.')
    parser.add_argument('--tqdm_format', default='{l_bar}{bar:20}{r_bar}{bar:-10b}', type=str,
                        help='Flag bar_format for the TQDM progress bar.')
    parser.add_argument('--disable_logs', action='store_true', help='Disables logging to the output log directory.')
    parser.add_argument('--refresh_brushes_file', action='store_true',
                        help='Rereads brush images and saves them to data/brushes.csv')

    # Simulation parameters
    # > Ground truth
    parser.add_argument('--theta_bg_intensity', default=[0, 0.6], type=float, nargs=2,
                        help='Bounds of the uniform distribution to draw background intensity.')
    parser.add_argument('--theta_n_akr', default=8, type=int,
                        help='Expected number of akr from the poisson distribution.')
    parser.add_argument('--theta_akr_intensity', default=[0, 0.15], type=float, nargs=2,
                        help='(Before absolute value) mean and std of AKR intensity.')
    # > Noise
    parser.add_argument('--theta_gaussian_intensity', default=[0.01, 0.04], type=float, nargs=2,
                        help='Bounds of the uniform distribution to determine the intensity of gaussian noise.')
    parser.add_argument('--theta_overall_channel_intensity', default=[0.3, 0.6], type=float, nargs=2,
                        help='Bounds of the uniform distribution to determine the overall intensity of channels.')
    parser.add_argument('--theta_n_channels', default=15, type=int,
                        help='Expected number of channels from the poisson distribution.')
    parser.add_argument('--theta_channel_height', default=4, type=int,
                        help='Expected *half* height of the channel from the exponential distribution.')
    parser.add_argument('--theta_channel_intensity', default=[0.1, 0.8], type=float, nargs=2,
                        help='Bounds of the uniform distribution to determine the individual intensity of channels.')
    # > Simulation scaling
    parser.add_argument('--disable_dataset_scaling', action='store_true',
                        help='Disables scaling of synthetic AKR in the dataset.')
    parser.add_argument('--dataset_intensity_scale', default=[0.2, 0.2], type=float, nargs=2,
                        help='Mean and standard deviation to scale the images to.')

    # Model parameters
    parser.add_argument('--img_size', default=[256, 384], type=int, nargs=2, help='Input size to DAARE.')
    parser.add_argument('--n_cdae', default=6, type=int,
                        help='The number of stacked convolutional denoising autoencoders in DAARE.')
    parser.add_argument('--depth', default=8, type=int, help='Depth of each convolutional denoising autoencoder.')
    parser.add_argument('--n_hidden', default=8, type=int, help='Size of each hidden Conv2d layer.')
    parser.add_argument('--kernel', default=[13, 5], type=int, nargs=2,
                        help='Kernel shape for the convolutional layers.')
    parser.add_argument('--n_norm', default=3, type=int,
                        help='The first n convolutional autoencoders to apply layernorm to.')

    # Training parameters
    parser.add_argument('--n_train', default=4096, type=int,
                        help='The number of training samples that are included in the training set.')
    parser.add_argument('--n_valid', default=1024, type=int,
                        help='The number of validation samples that are included in the validation set.')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size of to use in training and validation.')
    parser.add_argument('--n_epochs_per_cdae', default=10, type=int,
                        help='The number of epochs to train each convolutional denoising autoencoder.')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='The learning rate of each convolutional denoising autoencoder.')

    # Read arguments
    args = parser.parse_args()
    args.img_size = tuple(args.img_size)
    args.kernel = tuple(args.kernel)

    # Assertions
    assert (args.n_cdae >= args.n_norm), 'Number of layernorms is larger than the number of CDAEs.'

    return args


if __name__ == '__main__':
    # Get Arguments
    args = get_args()
    start_training(args)

