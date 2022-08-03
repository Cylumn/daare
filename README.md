# Denoising Autoencoder for Auroral Radio Emissions
## Table of Contents
1. [About](#about)
2. [Approach](#approach)
3. [Usage](#usage)
4. [Command Line Arguments](#arguments)
5. [API Tutorial](#api)

## <a name="about"></a>About
The Denoising Autoencoder for Auroral Radio Emissions (DAARE) 
is a tool to remove Radio Frequency Interference (RFI) commonly emerging
as horizontal emission lines from time-frequency spectrograms. This tool was
built to denoise Auroral Kilometric Radiation (AKR) observations from
the South Pole Station.

This work was generously supported by National Science Foundation 
grant AST-1950348, conducted at the MIT Haystack Observatory REU 2022 by
Allen Chang, and advised by Mary Knapp.

## <a name="approach"></a>Approach
![DAARE approach](daare.png)

## <a name="usage"></a>Usage
1. Install required packages.
```
pip install -r requirements.txt
```
2. To train a new model, run [train.py](/train.py).
3. To use a pretrained model, use [api.py](/api.py).

## <a name="arguments"></a> Command Line Arguments
### [train.py](/train.py)
#### Paths
`--path_to_data`: Path to the data directory.

`--path_to_logs`: Path to the logs directory.

`--path_to_output`: Path to the output directory.

#### Run options
`--model_name`: Name of the model when logging and saving.

`--verbose`: Trains with debugging outputs and print statements.                        

`--tqdm_format`: Flag bar_format for the TQDM progress bar.                                  

`--disable_logs`: Disables logging to the output log directory.                               

`--refresh_brushes_file`: Rereads brush images and saves them to [data/brushes.csv](data/brushes.csv) 

#### Simulation parameters
`--theta_bg_intensity`: Bounds of the uniform distribution to draw background intensity.                      

`--theta_n_akr`: Expected number of akr from the Poisson distribution.                                 

`--theta_akr_intensity`: (Before absolute value) mean and std of AKR intensity.                                

`--theta_gaussian_intensity`: Bounds of the uniform distribution to determine the intensity of Gaussian noise.      

`--theta_overall_channel_intensity`: Bounds of the uniform distribution to determine the overall intensity of channels.    

`--theta_n_channels`: Expected number of channels from the Poisson distribution.                            

`--theta_channel_height`: Expected **half** height of the channel from the exponential distribution.            

`--theta_channel_intensity`: Bounds of the uniform distribution to determine the individual intensity of channels. 

`--disable_dataset_scaling`: Disables scaling of synthetic AKR in the dataset.                                     

`--dataset_intensity_scale`: Mean and standard deviation to scale the images to.

#### Model parameters
`--img_size`: Input size to DAARE.                                                 

`--n_cdae`: The number of stacked convolutional denoising autoencoders in DAARE. 

`--depth`: Depth of each convolutional denoising autoencoder.                   

`--n_hidden`: Size of each hidden conv2d layer.                                    

`--kernel`: Kernel shape for the convolutional layers.                           

`--n_norm`: The first n convolutional autoencoders to apply layernorm to.        

#### Optimization
`--device_ids`: Device ids of the GPUs, if GPUs are available.                            

`--n_train`: The number of training samples that are included in the training set.     

`--n_valid`: The number of validation samples that are included in the validation set. 

`--batch_size`: Batch size of to use in training and validation.                          

`--n_epochs_per_cdae`: The number of epochs to train each convolutional denoising autoencoder.   

`--learning_rate`: The learning rate of each convolutional denoising autoencoder.

## <a name="api"></a>API Tutorial
