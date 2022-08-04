"""
The API was developed to load and run a pretrained model
without needing to have a prior understanding of PyTorch.
It is also meant to enable easy reading of
radio data written to disk and spectrogram generation.
Author: Allen Chang
Date Created: 08/03/2022
"""

import cv2
import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from typing import Union
import matplotlib.mlab as mlab

import digital_rf as drf

from model.daare import DAARE


class DAARE_API():
    def __init__(self,
                 path_to_model: str,
                 device: str = 'cpu',
                 tqdm_format: str = '{l_bar}{bar:20}{r_bar}{bar:-10b}'):
        """
        The initialization function for the DAARE API.
        :param path_to_model: Path to the model dictionary.
        :param device: The device to run on.
        :param tqdm_format: The bar_format flag for tqdm.
        """
        super(DAARE_API, self)

        self.path_to_model = path_to_model
        self.device = device
        self.tqdm_format = tqdm_format

        self.load_model(self.path_to_model)

    def load_model(self,
                   path_to_model: str):
        """
        Loads a pretrained model with args defined in the dictionary.
        :param path_to_model: Path to the model dictionary.
        :return: The torch DataParallel containing the model.
        """
        dict_model = torch.load(f'{path_to_model}', map_location=torch.device('cpu'))
        args = dict_model['args']

        model = nn.DataParallel(DAARE(depth=args.depth,
                                      hidden_channels=args.n_hidden,
                                      kernel=args.kernel,
                                      norm=True,
                                      img_size=args.img_size))
        for i in range(args.n_cdae):
            model.module.add_cdae(residual=(i > 0), norm=(i < args.n_norm))

        model.load_state_dict(dict_model['state_dict'])
        model.to(self.device)
        self.daare = model
        self.img_size = args.img_size
        self.intensity_scale = args.dataset_intensity_scale

    def read_drf(self,
                 files: Union[str, list],
                 channels: Union[str, list] = 'ant0',
                 n_fft: int = 1024,
                 n_bins: int = 1536,
                 n_samples: int = 1e5,
                 xmins: Union[float, list] = 0.0,
                 xmaxs: Union[float, list] = 1.0,
                 verbose: bool = False):
        """
        Reads a digital rf file as a spectrogram on a log-scale intensity.
        :param files: A list of paths to the digital_rf file.
        :param channels: A string or list of channels for each file.
        :param n_fft: The number of fast-Fourier transforms to produce the spectrogram.
        :param n_bins: The number of time bins to produce the spectrogram.
        :param n_samples: The number of samples to read to produce each bin.
        :param xmins: A float or list of the lower time bounds as proportion(s) to use for the spectrogram.
                      Accepts values between [0, 1].
        :param xmaxs: A float or list of the upper time bounds as proportion(s) to use for the spectrogram.
                      Accepts values between [0, 1].
        :param verbose: Print progress to console.
        :return: A tuple (spect, freq, time, start) of the spectrogram, frequency axis,
                 time axis, and starting unix time.
        """
        # Repeat
        if type(files) == str:
            files = [files]
        if type(channels) == str:
            channels = [channels for _ in files]
        if isinstance(xmins, Union[int, float].__args__):
            xmins = [xmins for _ in files]
        if isinstance(xmaxs, Union[int, float].__args__):
            xmaxs = [xmaxs for _ in files]

        # Assertions
        n_files = len(files)
        assert (len(channels) == n_files), "Length of channels must be equal to the length of files"
        assert (len(xmins) == n_files), "Length of xmins must be equal to the length of files"
        assert (len(xmaxs) == n_files), "Length of xmaxs must be equal to the length of files"
        for i in range(n_files):
            assert (0 <= xmins[i] <= 1), "xmin must be a float value within [0, 1]"
            assert (0 <= xmaxs[i] <= 1), "xmax must be a float value within [0, 1]"
            assert (xmins[i] < xmaxs[i]), "xmin must be lower than xmax"

        # Begin reading files
        spects = np.empty([n_files, n_fft, n_bins])
        freqs = np.empty([n_files, n_fft])
        times = np.empty([n_files, n_bins])
        starts = np.empty(n_files)
        for i in (range(n_files) if verbose else tqdm(range(n_files), desc='Files',
                                                      leave=False, position=0, bar_format=self.tqdm_format)):
            if verbose:
                print(f'> Reading file {i + 1} of {n_files}: {files[i]}')

            # Read in digital_rf and metadata
            drfObj = drf.DigitalRFReader(files[i])
            sps = drfObj.get_properties(channels[i])['samples_per_second']
            is_complex = drfObj.get_properties(channels[i])['is_complex']
            b = drfObj.get_bounds(channels[i])
            start = int((b[1] - b[0]) * xmins[i] + b[0])
            end = int((b[1] - b[0]) * xmaxs[i] + b[0])

            # Start Plotting
            spect = np.zeros([n_fft, n_bins], float)
            time = np.zeros([n_bins], int)
            stride = (end - start) // n_bins

            idx_start = start
            bin_range = np.arange(n_bins, dtype=np.int_)
            for idx_bin in (tqdm(bin_range, desc='Bins',
                                 leave=False, position=0, bar_format=self.tqdm_format) if verbose else bin_range):
                # Read drfObj
                data = drfObj.read_vector(idx_start,
                                          min(n_samples, stride, end - idx_start),
                                          channels[i])
                detrend_fn = mlab.detrend_mean
                psd_data, freq = mlab.psd(
                    data,
                    NFFT=(n_fft if is_complex else 2 * n_fft - 1),
                    Fs=float(sps),
                    detrend=detrend_fn,
                    scale_by_freq=False
                )

                # Log Scale
                spect[:, idx_bin] = np.real(10.0 * np.log10(np.abs(psd_data) + 1e-12))
                time[idx_bin] = idx_start / sps
                idx_start += stride

            spects[i] = spect
            freqs[i] = freq
            times[i] = time
            starts[i] = b[0] // sps

        return spects, freqs, times, starts

    def denoise(self,
                obs: np.array,
                batch_size: int = 16,
                retain_size: bool = False):
        """
        Denoises spectrogram(s) in batches.
        :param obs: A numpy array of observations.
        :param batch_size: The batch size to use for DAARE.
        :param retain_size: If true, returns a numpy array the same size as the input observation.
                            Otherwise, returns a numpy array with DAARE's output size.
        :return: A numpy array of denoised spectrograms
        """
        # Repeats
        if len(obs.shape) == 2:  # Single image
            obs = obs[None, None, :, :]
        elif len(obs.shape) == 3:  # Single image with batch
            obs = obs[:, None, :, :]

        # Assertions
        assert (len(obs.shape) == 4), "Invalid obs shape. obs must be of shape (height, width) for a single " \
                                      "image or (length, height, width) or " \
                                      "(length, 1, height, width) for a batch of images"
        assert (obs.shape[1] == 1), "Invalid number of channels. Expected a 1 channel image, but received " \
                                    "multiple channels. obs must be of shape (height, width) for a single " \
                                    "image or (length, height, width) or " \
                                    "(length, 1, height, width) for a batch of images"

        # Resize
        in_shape = obs.shape[-2:]
        obs_resized = np.empty([len(obs), 1, self.img_size[0], self.img_size[1]])
        # Check if need to resize
        if obs.shape[-2:] == obs_resized.shape[-2:]:
            obs_resized = obs
        else:
            # Resize each image
            for i in range(len(obs)):
                obs_resized[i][0] = cv2.resize(obs[i][0],
                                               dsize=(self.img_size[1], self.img_size[0]),
                                               interpolation=cv2.INTER_NEAREST)

        # Flatten
        obs_resized = obs_resized.reshape(-1, self.img_size[0] * self.img_size[1])
        # Standard scale
        means = obs_resized.mean(axis=1)[:, None].repeat(obs_resized.shape[1], axis=1)
        stds = obs_resized.std(axis=1)[:, None].repeat(obs_resized.shape[1], axis=1)
        obs_resized = (obs_resized - means) / stds
        # Scale to args
        obs_resized = (obs_resized * self.intensity_scale[1]) + self.intensity_scale[0]
        # Cutoffs
        obs_resized[obs_resized < 0] = 0
        obs_resized[obs_resized > 1] = 1
        # Reshape
        obs_resized = obs_resized.reshape(-1, 1, self.img_size[0], self.img_size[1])

        # Predict
        self.daare.eval()
        num_batches = int(np.ceil(len(obs) / batch_size))
        obs_resized = torch.tensor(obs_resized).to(self.device).float()
        obs_denoised = np.empty([len(obs), 1, self.img_size[0], self.img_size[1]])
        for idx_batch in tqdm(range(num_batches), leave=False, position=0, bar_format=self.tqdm_format):
            x = obs_resized[idx_batch:idx_batch + batch_size]
            obs_denoised[idx_batch:idx_batch + batch_size] = self.daare(x).detach().numpy()

        # Rescale
        obs_denoised = obs_denoised.reshape(-1, self.img_size[0] * self.img_size[1])
        obs_denoised = (obs_denoised - self.intensity_scale[0]) / self.intensity_scale[1]
        obs_denoised = (obs_denoised * stds) + means
        obs_denoised = obs_denoised.reshape(-1, self.img_size[0], self.img_size[1])

        # Resize
        if retain_size and in_shape != obs_denoised.shape[-2:]:
            obs_resized = np.empty([len(obs), in_shape[0], in_shape[1]])
            for i in range(len(obs)):
                obs_resized[i] = cv2.resize(obs_denoised[i],
                                            dsize=(in_shape[1], in_shape[0]),
                                            interpolation=cv2.INTER_NEAREST)
            obs_denoised = obs_resized

        return obs_denoised
