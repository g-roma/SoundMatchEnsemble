#!/usr/bin/env python3
# compare neural network architectures
import numpy as np
from common.models import MLPModel, CNNModel, LSTMModel
from common.data  import get_loaders
from common.train import train_model
from common.evaluation import test_model
from common.config import device

basic_synth = "1022210"
complex_synth = "1111111"

# modify for different experiments
synth_id = complex_synth
#synth_id = basic_synth

synth_path = "path/to/ensemble/datasets/" + synth_id

x = np.load(synth_path + "/mfcc/0.mfcc.npy")
H, W = x.shape
y = np.genfromtxt(synth_path + "/labels/0.csv", 
	delimiter = ",", dtype = np.float64, usemask = False)
O = y.shape[0] - 1

# Modify for different models
#model = MLPModel(H*W, O).to(device)
model = CNNModel(O).to(device)
#model = LSTMModel(20, 40, O, 1).to(device)

model.train()
train_data, val_data, test_data = get_loaders(synth_path)
train_time = train_model(model, train_data, val_data)
(param, spec) = test_model(model, test_data, synth_path, synth_id)
print(param, spec, train_time)


