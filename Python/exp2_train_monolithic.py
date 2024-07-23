#!/usr/bin/env python3
# train programmer with monolithic dataset
import torch
import numpy as np
from common.models import CNNModel
from common.data  import get_loaders
from common.train import train_model
from common.config import device

model_name = "exp2_monolithic.pt"
synth_path = "/path/to/monolithic/dataset"
x = np.load(synth_path + "/mfcc/0.mfcc.npy")
H, W = x.shape
y = np.genfromtxt(synth_path + "/labels/0.csv", 
	delimiter=",", dtype = np.float64, usemask = False)
O = y.shape[0] - 1
model = CNNModel(O).to(device)
model.train()
train_data, val_data, test_data = get_loaders(synth_path)
train_time = train_model(model, train_data, val_data)
torch.save(model.state_dict(), model_name)