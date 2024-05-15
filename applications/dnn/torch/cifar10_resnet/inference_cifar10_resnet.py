"""
Parameterizable inference simulation script for CIFAR-10 ResNets.
"""

import torch
from torchvision import datasets, transforms
import numpy as np
import warnings, sys, time
from build_resnet_cifar10 import ResNet_cifar10
warnings.filterwarnings('ignore')
sys.path.append("../") # to import dnn_inference_params
sys.path.append("../../../../") # to import simulator
from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, convertible_modules, reinitialize
from find_adc_range import find_adc_range
from dnn_inference_params import dnn_inference_params

## Depth parameter for model selection
# Follows definition in original ResNet paper (He et al, CVPR 2016)
# n = 2 : ResNet-14 (175K weights)
# n = 3 : ResNet-20 (272K weights)
# n = 5 : ResNet-32 (467K weights)
# n = 9 : ResNet-56 (856K weights)
n = 3

useGPU = True # use GPU?
N = 10000 # number of images
batch_size = 64
Nruns = 1
print_progress = True

depth = 6*n+2
print("Model: ResNet-{:d}".format(depth))
print("CIFAR-10: using "+("GPU" if useGPU else "CPU"))
print("Number of images: {:d}".format(N))
print("Number of runs: {:d}".format(Nruns))
print("Batch size: {:d}".format(batch_size))
device = torch.device("cuda:0" if (torch.cuda.is_available() and useGPU) else "cpu")

##### Load Pytorch model
resnet_model = ResNet_cifar10(n)
resnet_model = resnet_model.to(device)
resnet_model.load_state_dict(
    torch.load('./models/resnet{:d}_cifar10.pth'.format(depth),
    map_location=torch.device(device)))
resnet_model.eval()
n_layers = len(convertible_modules(resnet_model))

##### Set the simulation parameters

# Create a list of CrossSimParameters objects
params_list = [None] * n_layers

# Params arguments common to all layers
base_params_args = {
    'ideal' : False,
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
    'alpha_error' : 0.0,
    'proportional_error' : False,
    'noise_model' : "none",
    'alpha_noise' : 0.0,
    'proportional_noise' : False,
    'drift_model' : "none",
    't_drift' : 0,
    ## Array properties
    'NrowsMax' : 1152,
    'NcolsMax' : None,
    'Rp_row' : 0, # ohms
    'Rp_col' : 0, # ohms
    'interleaved_posneg' : False,
    'subtract_current_in_xbar' : True,
    'gate_input' : False,
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
    'useGPU' : useGPU,
    'conv_matmul' : True,
    }

### Load input limits
input_ranges = np.load("./calibrated_config/input_limits_ResNet{:d}.npy".format(depth))

### Load ADC limits
adc_ranges = find_adc_range(base_params_args, n_layers, depth)

### Load (x_par, y_par) values
xy_pars = np.load("./calibrated_config/resnet{:d}_xy.npy".format(depth))
if base_params_args['Rp_row'] > 0 or base_params_args['Rp_col'] > 0:
    xy_pars = np.load("./calibrated_config/resnet{:d}_xy_parasitics.npy".format(depth))

### Set the parameters
for k in range(n_layers):
    params_args_k = base_params_args.copy()
    params_args_k['positiveInputsOnly'] = (False if k == 0 else True)
    params_args_k['input_range'] = input_ranges[k]
    params_args_k['adc_range'] = adc_ranges[k]
    params_args_k['x_par'] = xy_pars[k,0]
    params_args_k['y_par'] = xy_pars[k,1]
    params_list[k] = dnn_inference_params(**params_args_k)

#### Convert PyTorch layers to analog layers
analog_resnet = from_torch(resnet_model, params_list, fuse_batchnorm=True, bias_rows=0)

#### Load and transform CIFAR-10 dataset
normalize = transforms.Normalize(
    mean = [0.485, 0.456, 0.406],
    std  = [0.229, 0.224, 0.225])
dataset = datasets.CIFAR10(root='./',train=False, download=True, 
    transform= transforms.Compose([transforms.ToTensor(), normalize]))
dataset = torch.utils.data.Subset(dataset, np.arange(N))
cifar10_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

#### Run inference and evaluate accuracy
accuracies = np.zeros(Nruns)
for m in range(Nruns):

    T1 = time.time()
    y_pred, y, k = np.zeros(N), np.zeros(N), 0
    for inputs, labels in cifar10_dataloader:
        inputs = inputs.to(device)
        output = analog_resnet(inputs)
        output = output.to(device)
        y_pred_k = output.data.cpu().detach().numpy()
        if batch_size == 1:
            y_pred[k] = y_pred_k.argmax()
            y[k] = labels.cpu().detach().numpy()
            k += 1
        else:
            batch_size_k = y_pred_k.shape[0]
            y_pred[k:(k+batch_size_k)] = y_pred_k.argmax(axis=1)
            y[k:(k+batch_size_k)] = labels.cpu().detach().numpy()
            k += batch_size_k
        if print_progress:
            print("Image {:d}/{:d}, accuracy so far = {:.2f}%".format(
                k, N, 100*np.sum(y[:k] == y_pred[:k])/k), end="\r")
    
    T2 = time.time()
    top1 = np.sum(y == y_pred)/len(y)
    accuracies[m] = top1
    print("\nInference finished. Elapsed time: {:.3f} sec".format(T2-T1))
    print('Accuracy: {:.2f}% ({:d}/{:d})\n'.format(top1*100,int(top1*N),N))
    if m < (Nruns - 1):
        reinitialize(analog_resnet)

if Nruns > 1:
    print("==========")
    print("Mean accuracy:  {:.2f}%".format(100*np.mean(accuracies)))
    print("Stdev accuracy: {:.2f}%".format(100*np.std(accuracies)))