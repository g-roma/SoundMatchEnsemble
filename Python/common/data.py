import numpy as np
import os, glob
from torch.utils.data import DataLoader, Subset
from .datasets import MFCCDataset

def get_loaders(db_path, train_frac = 0.8, val_frac = 0.1, test_frac = 0.1):
    if (train_frac + val_frac + test_frac) != 1: 
        print("ERROR: wrong split")
    ds = MFCCDataset(db_path)
    num_samples = len(ds)
    num_val = int(num_samples * val_frac)
    num_test = int(num_samples * test_frac)
    num_train = int(num_samples * train_frac)
    print("Data split:", num_train, num_val, num_test)

    train_set = Subset(ds, 
        range(1, num_train))
    val_set = Subset(ds, 
        range(num_train, num_train + num_val))
    test_set = Subset(ds, 
        range(num_train + num_val,num_train + num_val + num_test))
    
    train_loader = DataLoader(train_set, batch_size = 250, shuffle = True)
    val_loader = DataLoader(val_set, batch_size = 250,  shuffle = True)
    test_loader = DataLoader(test_set, batch_size = 250)
    return (train_loader, val_loader, test_loader )

def get_test_data(path):
    files = glob.glob(path + "/mfcc/*.mfcc.npy")
    test_data = [np.load(f) for f in files]
    filenames = [os.path.basename(f) for f in files]
    return np.stack(test_data), filenames

    