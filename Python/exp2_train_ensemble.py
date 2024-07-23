#!/usr/bin/env python3
# train ensemble models
import torch
import numpy as np
from common.models import CNNModel
from common.data  import get_loaders
from common.train import train_model
from common.config import ensemble_synth_ids, device

model_name = "exp2_ensemble_model.pt"
base_path = "path/to/ensemble/datasets"

for s in ensemble_synth_ids:
    print(s)
    synth_path = base_path + s
    x = np.load(synth_path + "/mfcc/0.mfcc.npy")
    H, W = x.shape
    y = np.genfromtxt(synth_path + "/labels/0.csv", 
        delimiter = ",", dtype = np.float64, usemask = False)
    O = y.shape[0] - 1
    model = CNNModel(O).to(device)
    model.train()
    train_data, val_data, test_data = get_loaders(synth_path)
    train_time = train_model(model, train_data, val_data)
    torch.save(model.state_dict(), synth_path + "/" + model_name)
    print("---------------")

