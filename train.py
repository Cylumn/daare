'''
Training script for DAARE
Author: Allen Chang
Date Created: 08/02/2022
'''

import argparse

from data import simulate


def get_args():
    parser = argparse.ArgumentParser('DAARE', add_help=False)

    # Paths
    parser.add_argument('--path_to_data', default='data', type=str, help='Path to the data directory.')
    parser.add_argument('--path_to_logs', default='logs', type=str, help='Path to the logs directory.')

    # Options
    parser.add_argument('--model_name', default='daare_v1', type=str, help='Name of the model when logging and saving.')
    parser.add_argument('--verbose', action='store_true', help='Trains with debugging outputs and print statements.')
    parser.add_argument('--disable_logs', action='store_false', help='Disables logging to the output log directory.')
    parser.add_argument('--refresh_brushes_file', action='store_true',
                        help='Rereads brush images and saves them to data/brushes.csv')

    # Simulation parameters
    parser.add_argument('--theta_bg_intensity', default=[0, 0.6], type=float, nargs=2,
                        help='Bounds of the uniform distribution to draw background intensity.')
    parser.add_argument('--theta_n_akr', default=8, type=int,
                        help='Expected number of akr from the poisson distribution.')
    parser.add_argument('--theta_akr_intensity', default=[0, 0.15], type=float, nargs=2,
                        help='(Before absolute value) mean and std of AKR intensity.')
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

    # Model parameters
    parser.add_argument('--img_size', default=[256, 384], type=int, nargs=2, help='Input size to DAARE.')
    parser.add_argument('--n_cdae', default=6, type=int,
                        help='The number of stacked convolutional denoising autoencoders in DAARE.')
    parser.add_argument('--depth', default=8, type=int, help='Depth of each convolutional denoising autoencoder.')
    parser.add_argument('--kernel', default=[13, 5], type=int, nargs=2,
                        help='Kernel shape for the convolutional layers.')
    parser.add_argument('--n_layernorm', default=3, type=int,
                        help='The first n convolutional autoencoders to apply layernorm to.')

    # Training parameters
    parser.add_argument('--n_train', default=4096, type=int,
                        help='The number of training samples that are included in the training set.')
    parser.add_argument('--n_valid', default=1024, type=int,
                        help='The number of validation samples that are included in the validation set.')
    parser.add_argument('--n_epochs_per_cdae', default=10, type=int,
                        help='The number of epochs to train each convolutional denoising autoencoder.')
    parser.add_argument('--learning_rate', default=1e-4, type=float,
                        help='The learning rate of each convolutional denoising autoencoder.')

    # Read arguments
    args = parser.parse_args()
    args.img_size = tuple(args.img_size)
    args.kernel = tuple(args.kernel)

    # Assertions
    assert (args.n_cdae >= args.n_layernorm), 'Number of layernorms is larger than the number of CDAEs.'

    return args


if __name__ == '__main__':
    # Get Arguments
    args = get_args()

    brushes = simulate.read_brushes(args)
    x = simulate.ground_truth(brushes, args)
    x = simulate.noise(x, args)
    print(x)
