"""
Parameterizable inference simulation script for MLPerf's ResNet50-v1.5
benchmark model on ImageNet.
"""

import torch
import numpy as np
import warnings, sys, time
from build_resnet50 import resnet50
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')
sys.path.append("../../") # to import dnn_inference_params
sys.path.append("../../../../") # to import simulator
from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules, reinitialize
from dnn_inference_params import dnn_inference_params
from find_adc_range import find_adc_range

useGPU = False # use GPU?
N = 1000 # number of images
batch_size = 32
Nruns = 1
print_progress = True

print("ImageNet: using "+("GPU" if useGPU else "CPU"))
print("Number of images: {:d}".format(N))
print("Number of runs: {:d}".format(Nruns))
print("Batch size: {:d}".format(batch_size))
device = torch.device("cuda:0" if (torch.cuda.is_available() and useGPU) else "cpu")

##### Load Pytorch model
resnet50_model = resnet50(pretrained=True).to(device)
resnet50_model.eval()
n_layers = len(convertible_modules(resnet50_model))

##### Set the simulation parameters

# Create a list of CrossSimParameters objects
params_list = [None] * n_layers

# Params arguments common to all layers
base_params_args = {
    'ideal' : True,
    ## Mapping style
    'core_style' : "BALANCED",
    'Nslices' : 1,
    ## Weight value representation and precision
    'weight_bits' : 8,
    'weight_percentile' : 100,
    'digital_bias' : True,
    ## Memory device
    'Rmin' : 1e4,
    'Rmax' : 1e6,
    'infinite_on_off_ratio' : False,
    'error_model' : "none",
    'alpha_error' : 0.00,
    'proportional_error' : False,
    'noise_model' : "none",
    'alpha_noise' : 0.00,
    'proportional_noise' : False,
    'drift_model' : "none",
    't_drift' : 0,
    ## Array properties
    'NrowsMax' : 1152, # inputs
    'NcolsMax' : 256, # outputs
    'Rp_row' : 0, # ohms
    'Rp_col' : 0, # ohms
    'interleaved_posneg' : False,
    'subtract_current_in_xbar' : True,
    'current_from_input' : True,
    ## Input quantization
    'input_bits' : 8,
    'input_bitslicing' : False,
    'input_slice_size' : 1,
    ## ADC
    'adc_bits' : 8,
    'adc_range_option' : "CALIBRATED",
    'adc_type' : "generic",
    'adc_per_ibit' : False,
    ## Simulation parameters
    'useGPU' : useGPU
    }

### Load input limits
input_ranges = np.load("./calibrated_config/dac_limits_ResNet50v15.npy")

### Load ADC limits
adc_ranges = find_adc_range(base_params_args, n_layers)

### Set the parameters
for k in range(n_layers):
    params_args_k = base_params_args.copy()
    params_args_k['positiveInputsOnly'] = (False if k == 0 else True)
    params_args_k['input_range'] = input_ranges[k]
    params_args_k['adc_range'] = adc_ranges[k]
    params_list[k] = dnn_inference_params(**params_args_k)

# Convert PyTorch layers to analog layers
analog_resnet50 = from_torch(resnet50_model, params_list, fuse_batchnorm=True, bias_rows=0)

#### Load pre-processed ImageNet dataset
# Make sure this comes after params are set since the params can affect batch size
imagenet_path = "../../../../../../imagenet/"
if N <= 1000:
    x = np.load(imagenet_path + "x_val_MLperfRN50_1000.npy")[:N,:,:,:]
elif N <= 25000:
    x = np.load(imagenet_path + "x_val_MLperfRN50_25000_front.npy")[:N,:,:,:]
x = x.transpose(0,3,1,2)
y = np.load(imagenet_path + "y_val.npy")[:N] + 1
imagenet_dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))
imagenet_dataloader = DataLoader(imagenet_dataset, batch_size=batch_size)

#### Run inference and evaluate accuracy

accuracies = np.zeros(Nruns)
for m in range(Nruns):
    T1 = time.time()
    y_pred, k = np.zeros(N), 0
    for inputs, labels in imagenet_dataloader:
        inputs = inputs.to(device)
        output = analog_resnet50(inputs)
        output = output.to(device)
        y_pred_k = output.data.cpu().detach().numpy()
        if batch_size == 1:
            y_pred[k] = y_pred_k.argmax()
            k += 1
        else:
            batch_size_k = y_pred_k.shape[0]
            y_pred[k:(k+batch_size_k)] = y_pred_k.argmax(axis=1)
            k += batch_size_k
        if print_progress:
            print("Image {:d}/{:d}, top-1 accuracy so far = {:.2f}%".format(
                k, N, 100*np.sum(y[:k] == y_pred[:k])/k), end="\r")
    T2 = time.time()
    top1 = np.sum(y == y_pred)/len(y)
    accuracies[m] = top1
    print("\nInference finished. Elapsed time: {:.3f} sec".format(T2-T1))
    print('Top-1 accuracy: {:.3f}% ({:d}/{:d})\n'.format(top1*100,int(top1*N),N))
    if m < (Nruns - 1):
        reinitialize(analog_resnet50)

if Nruns > 1:
    print("==========")
    print("Mean accuracy:  {:.3f}%".format(100*np.mean(accuracies)))
    print("Stdev accuracy: {:.3f}%".format(100*np.std(accuracies)))