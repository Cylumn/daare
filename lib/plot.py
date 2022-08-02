"""
Helper file to plot spectrograms.
Author: Allen Chang
Date Created: 08/02/2022
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def spect_simple(data,
                 frequency=None,
                 time=None,
                 start=None,
                 cmap='jet',
                 dpi=400,
                 fs=4,
                 vmin=0,
                 vmax=1):
    """
    Visualizes a simple spectrogram.

    :param data: Input spectrogram.
    :param frequency: An array of tick values for the frequency axis of the spectrogram.
    :param time: An array of the unix time values for the time axis of the spectrogram.
    :param start: Starting unix time for the title of the spectrogram.
    :param cmap: Colormap to use for the spectrogram.
    :param dpi: Resolution of the figure.
    :param fs: Font size.
    :param vmin: Lower value bound of the cmap.
    :param vmax: Upper value bound of the cmap.
    :return: Matplotlib axis object.
    """
    # Font to use
    font = {'family': 'DejaVu Sans', 'weight': 'normal', 'size': fs}
    matplotlib.rc('font', **font)

    # Create plot
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    spect = [np.arange(0, 1e6, 1e6 / data.shape[0]), np.arange(0, 1e3, 1e3 / data.shape[1]), data]
    im = ax.imshow(spect[2], cmap=cmap, origin='lower', vmin=vmin, vmax=vmax, interpolation="nearest")
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    # Axis and Labeling
    if frequency is not None:
        ax.set_yticks(np.arange(0, len(frequency), len(frequency) // 8))
        ax.set_yticklabels(frequency[::len(frequency) // 8])
    if time is not None:
        import time as t
        from datetime import datetime

        assert (start is not None), "Time should not be plotted without a starting time."

        start_struct = t.gmtime(int(start))
        x_ticks = np.arange(0, len(time), len(time) // 5)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([datetime.utcfromtimestamp(np.real(x)).strftime('%H:%M:%S') for x in time[x_ticks]])
        ax.set_title(t.strftime('%Y-%m-%d %H:%M:%S UTC', start_struct), fontsize=5)

    return ax


def spects(rows,
           nrows_ncols):
    """
    Plots multiple spectrograms in an ImageGrid.
    :param rows: List of tensors to plot.
    :param nrows_ncols: Number of rows, number of columns to plot.
    """
    import torch
    from mpl_toolkits.axes_grid1 import ImageGrid

    # Create figure
    fig = plt.figure(figsize=(20., 20.))
    rows = [(col.detach() if type(col) == torch.Tensor else torch.tensor(col)) for col in rows]
    img_shape = rows[0].shape[-2:]
    axis = 0 if (nrows_ncols[0] == 1 or nrows_ncols[1] == 1) else 1
    rows = torch.cat(rows, axis=axis).view(nrows_ncols[0], nrows_ncols[1], img_shape[0], img_shape[1])
    grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.1)

    # Place the images on the grid
    df_grid = rows.reshape(-1, rows.shape[-2], rows.shape[-1])
    for ax, im in zip(grid, df_grid):
        if len(im.shape) > 2:
            im = im.reshape(im.shape[-2:])
        ax.imshow(im, origin='lower', cmap='jet', vmin=0, vmax=1)

    plt.show()
