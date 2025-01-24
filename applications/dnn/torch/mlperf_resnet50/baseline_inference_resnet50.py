"""
Baseline PyTorch inference script for MLPerf's ResNet50-v1.5
benchmark model on ImageNet, for comparison with CrossSim simulation
results.
"""

import torch
import numpy as np
import warnings, sys, time
from build_resnet50 import resnet50
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')

N = 1000 # number of images
useGPU = True # use GPU?
batch_size = 32

print("ImageNet: using "+("GPU" if useGPU else "CPU"))
print("Number of images: {:d}".format(N))
print("Batch size: {:d}".format(batch_size))
device = torch.device("cuda:0" if (torch.cuda.is_available() and useGPU) else "cpu")

# Load pre-processed ImageNet dataset
imagenet_path = "../../../../../../imagenet/"
if N <= 25000:
    x = np.load(imagenet_path + "x_val_MLperfRN50_25000_front.npy")[:N,:,:,:].transpose(0,3,1,2)
    y = np.load(imagenet_path + "y_val.npy")[:N] + 1
else:
    x = np.zeros((N, 3, 224, 224))
    x[:25000,:,:,:] = np.load(imagenet_path + "x_val_MLperfRN50_25000_front.npy").transpose(0,3,1,2)
    x[25000:N,:,:,:] = np.load(imagenet_path + "x_val_MLperfRN50_25000_back.npy")[:(N-25000),:,:,:].transpose(0,3,1,2)
    y = np.load(imagenet_path + "y_val.npy")[:N] + 1
imagenet_dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y)) # create your datset
imagenet_dataloader = DataLoader(imagenet_dataset, batch_size=batch_size) # create your dataloader

# Load Pytorch model
resnet50_model = resnet50(pretrained=True).to(device)
resnet50_model.eval()

# Evaluate accuracy of Pytorch model
y_pred, k = np.zeros(N), 0
TA = time.time()
for inputs, labels in imagenet_dataloader:
    inputs = inputs.to(device)
    output = resnet50_model(inputs)
    output = output.to(device)
    y_pred_k = output.data.cpu().detach().numpy()
    batch_size_k = y_pred_k.shape[0]
    y_pred[k:(k+batch_size_k)] = y_pred_k.argmax(axis=1)
    k += batch_size_k

TB = time.time()
top1 = np.sum(y == y_pred)/len(y)
print("\n== Baseline Pytorch evaluation ==")
print("Inference finished. Elapsed time: {:.3f} sec".format(TB-TA))
print('Top-1 accuracy: {:.3f}% ({:d}/{:d})'.format(top1*100,int(top1*N),N))
