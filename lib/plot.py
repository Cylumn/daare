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
                 vmin=None,
                 vmax=None):
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

    # Set vmin, vmax
    if vmin is None:
        data_flattened = data.flatten()
        vmin = data_flattened.mean() - data_flattened.std()
    if vmax is None:
        data_flattened = data.flatten()
        vmax = data_flattened.mean() + data_flattened.std() * 4

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
           nrows_ncols,
           cbar_col: int = None):
    """
    Plots multiple spectrograms in an ImageGrid.
    :param rows: List of tensors to plot.
    :param nrows_ncols: Number of rows, number of columns to plot.
    :param cbar_col: The index of the column that changes the extent of the colorbar.
                     If 'None' is passed, uses all the columns.
    """
    import torch
    from mpl_toolkits.axes_grid1 import ImageGrid

    # Assertions
    assert (nrows_ncols[0] > 0 and nrows_ncols[1] > 0), "Rows and columns must be greater than 0"
    assert (cbar_col >= 0 and cbar_col < nrows_ncols[1]), "cbar_col must be between [0, n_cols)"

    # Reshape rows
    rows = [(col.detach() if type(col) == torch.Tensor else torch.tensor(col)) for col in rows]
    img_size = rows[0].shape[-2:]
    axis = 0 if (nrows_ncols[0] == 1 or nrows_ncols[1] == 1) else 1
    rows = torch.cat(rows, axis=axis).view(nrows_ncols[0], nrows_ncols[1], img_size[0], img_size[1])

    # Make ImageGrid
    fig = plt.figure(figsize=(20., 20.))
    grid = ImageGrid(fig, 111, nrows_ncols=nrows_ncols, axes_pad=0.1,
                     cbar_location="right",
                     cbar_mode="edge",
                     cbar_size="5%",
                     cbar_pad="2%")

    # Calculate row-wise vmin, vmax
    vmins = []
    vmaxs = []
    for row in rows:
        if cbar_col is None:
            row_flattened = row.flatten()
        else:
            row_flattened = row[cbar_col].flatten()
        vmins.append(row_flattened.mean() - row_flattened.std())
        vmaxs.append(row_flattened.mean() + row_flattened.std() * 4)

    # Place the images on the grid
    df_grid = rows.reshape(-1, rows.shape[-2], rows.shape[-1])
    for i in range(len(grid)):
        # Image
        x = df_grid[i]
        # If has channel dimension
        if len(x.shape) > 2:
            x = x.reshape(x.shape[-2:])
        # Imshow with vmin, vmax
        im = grid[i].imshow(x, origin='lower', cmap='jet',
                            vmin=vmins[i // nrows_ncols[1]], vmax=vmaxs[i // nrows_ncols[1]])
        # Plot colorbar
        plt.colorbar(im, cax=grid.cbar_axes[i], orientation='vertical')

    return grid
