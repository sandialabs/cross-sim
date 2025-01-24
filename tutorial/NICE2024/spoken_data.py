# %%
import sys
import os
datapath = os.path.expanduser('~/free-spoken-digit-dataset')
sys.path.append(datapath + '/utils')
from fsdd import FSDD
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# %%
class Heidelberg_Dataset(Dataset):
    def __init__(self, root= '~/free-spoken-digit-dataset'):

        root = os.path.expanduser(root)
        sys.path.append(root)
        sys.path.append(root + '/utils')
        spectrogram_dir =  root + '/spectrograms'
        import fsdd
        m = fsdd.FSDD(root)
        a = m.get_spectrograms(spectrogram_dir)

        x_ = list()
        y_ = list()
        
        for x, y, _ in iter(a):
            x = torch.from_numpy(np.asarray(x).mean(axis=-1)).unsqueeze(0)
            y = torch.nn.functional.one_hot(torch.from_numpy(np.asarray(int(y))), 10)
            x_.append(x)
            y_.append(y)

        x = torch.stack(x_).float()
        x = x.squeeze()
        x = x.transpose(1,2)
        x = x.unsqueeze(2)
        self.x = x / 256

        y = torch.stack(y_).unsqueeze(1).float()
        y = y.repeat(1,64,1)
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
def get_loaders(batch_size=32):
    dataset = Heidelberg_Dataset()
    ds_train, ds_val = random_split(dataset, [0.8, 0.2])

    loader_train = DataLoader(ds_train, 
                          batch_size=batch_size,
                          shuffle=True)

    loader_val = DataLoader(ds_val,
                        batch_size=batch_size,
                        shuffle=False)
    
    return loader_train, loader_val