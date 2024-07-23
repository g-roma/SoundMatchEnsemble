#!/usr/bin/env python3
# evaluation of monolithic synth programmer
import numpy as np
import torch
from common.models import CNNModel
from common.data import get_test_data
from common.config import device, nsynth_valid
from common.evaluation import get_mfcc_dists

model_name = "exp2_monolithic.pt"
data_path = "/tmp/"

def eval_monolithic(test_data, name):
    out_size = 19
    mono_model = CNNModel(out_size).to(device)
    mono_model.load_state_dict(torch.load(model_name))
    mfcc = torch.from_numpy(test_data).to(device)
    predicted = mono_model(mfcc)
    mono_errs = get_mfcc_dists(mfcc, predicted, data_path, None)
    np.savetxt(f"{name}_results.csv", mono_errs, delimiter = ",")
    return mono_errs

test_data, filenames = get_test_data(nsynth_valid)
result = eval_monolithic(test_data, "monolithic")
print("--------------")
print(np.median(result))

