"""
The Denoising Autoencoder for Auroral Radio Emissions (DAARE) architecture.
Author: Allen Chang
Date Created: 08/02/2022
"""
import torch
from torch import nn


class CDAEComponent(nn.Module):
    """
    A Convolutional Denoising AutoEncoder Component.
    """
    def __init__(self,
                 depth: int,
                 hidden_channels: int,
                 kernel: tuple,
                 padding: tuple,
                 norm: bool,
                 img_size: tuple,
                 in_channels: int,
                 out_channels: int):
        """
        Initializing function for the CDAE Component.
        :param depth: The number of conv2d layers that the CDAE uses.
        :param hidden_channels: The number of channels for hidden conv2d layers.
        :param kernel: The kernel size of the conv2d layers.
        :param padding: The padding size of the conv2d layers.
        :param norm: Whether to include layernorm or not in this CDAE.
        :param img_size: The image size of the input images.
        :param in_channels: The number of channels to input to the CDAE.
        :param out_channels: The number of channels that the CDAE outputs.
        """
        super(CDAEComponent, self).__init__()

        self.img_size = img_size
        self.blocks = nn.ModuleList()
        for d in range(depth):
            in_size = hidden_channels if d > 0 else in_channels
            out_size = hidden_channels if d < depth - 1 else out_channels
            norm = (d < d - 1) and norm  # Last layer has no Layer Norm
            tanh = (d == depth - 1)  # Tanh only on final layer
            self.blocks.append(self.block(in_size, out_size, kernel, padding, norm, tanh))

    def block(self,
              in_size: int,
              out_size: int,
              kernel: tuple,
              padding: tuple,
              norm: bool,
              tanh: bool):
        """
        Creates a conv2d-*-** block with an optional layernorm and leakyrelu/tanh.
        :param in_size: Input size of the conv2d.
        :param out_size: Output size of the conv2d.
        :param kernel: Kernel size of the conv2d.
        :param padding: Padding size of the conv2d.
        :param norm: Whether to use layernorm or not.
        :param tanh: Activation at the end of the block.
        :return: Sequential object of the block.
        """
        layers = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=kernel, stride=1, padding=padding))
        if norm:
            layers.append(nn.LayerNorm([out_size, self.img_size[0], self.img_size[1]]))
        if tanh:
            layers.append(nn.Tanh())
        else:
            layers.append(nn.LeakyReLU(0.2))

        return layers

    def forward(self,
                x: torch.Tensor):
        """
        Forward-propagates the input tensor using the blocks.
        """
        for block in self.blocks:
            x = block(x)
        return x


class DAARE(nn.Module):
    """
    The Denoising AutoEncoder for Auroral Radio Emissions (DAARE) model.
    """
    def __init__(self,
                 depth: int,
                 hidden_channels: int,
                 kernel: tuple,
                 norm: bool,
                 img_size: tuple,
                 in_channels: int = 1,
                 out_channels: int = 1):
        """
        Initializing function for DAARE.
        :param depth: The number of conv2d layers that the CDAE uses.
        :param hidden_channels: The number of channels for hidden conv2d layers.
        :param kernel: The default kernel size of the conv2d layers.
        :param norm: The default on whether to include layernorm or not in this CDAE.
        :param img_size: The image size of the input images.
        :param in_channels: The number of channels to input to the CDAE.
        :param out_channels: The number of channels that the CDAE outputs.
        """
        super(DAARE, self).__init__()

        self.depth = depth
        self.hidden_channels = hidden_channels
        self.kernel = kernel
        self.padding = (kernel[0] // 2, kernel[1] // 2)
        self.norm = norm
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Add the first component
        self.components = nn.ModuleList()

    def add_cdae(self,
                 residual: bool = True,
                 kernel: tuple = None,
                 norm: bool = None):
        """
        Adds a CDAE component to the current stack.
        :param residual: Whether to include an additional skip connection from the input image.
        :param kernel: The kernel size of the CDAE.
        :param norm: Whether to use layernorm or not in this CDAE.
        :return:
        """
        # Compute padding and defaults
        if kernel is None:
            kernel = self.kernel
            padding = self.padding
        else:
            padding = (kernel[0] // 2, kernel[1] // 2)
        if norm is None:
            norm = self.norm

        # Freeze previous components
        for component in self.components[:-1]:
            for param in component.parameters():
                param.requires_grad = False

        # Add new component
        self.components.append(CDAEComponent(self.depth, self.hidden_channels,
                                             kernel, padding, norm, self.img_size,
                                             self.in_channels + self.out_channels if residual else self.in_channels,
                                             self.out_channels))

    def forward(self,
                x: torch.Tensor,
                n_components: int = None,
                return_intermediate: bool = False):
        """
        Forward-propagates across stacked CDAEs.
        :param x: Input AKR observation
        :param n_components: Number of CDAE components to use in the prediction.
        :param return_intermediate: Flag to return intermediately denoised AKR instead of the entire difference.
                                    This flag is used to calculate MSE loss of noise.
        :return: A detached tensor containing the denoised spectrogram.
        """
        # Default to using all components
        if not n_components:
            n_components = len(self.components)

        assert (n_components > 0), "At least one CDAE component must be used to predict denoised AKR."

        # Input component
        x_inter = x
        z_inter = self.components[0](x)
        # Rest of the components
        for component in self.components[1:n_components]:
            x_inter = x_inter - z_inter
            z_inter = component(torch.cat([x, x_inter], axis=1))

        if return_intermediate:
            return x_inter, z_inter
        else:
            # Return difference of incremental observation and incremental noise
            return (x_inter - z_inter).detach()