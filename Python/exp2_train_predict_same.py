#!/usr/bin/env python3
# train "predict same" classifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from common.models import CNNModel
from common.config import device, ensemble_synth_ids
from common.datasets import PredictSameDataset
from common.train import train_model
from common.evaluation import get_accuracy

base_path = "/path/to/ensemble/datasets"
model_name = "exp2_predict_same.pt"

train_ds = PredictSameDataset(base_path, ensemble_synth_ids, 0, 1900)
val_ds = PredictSameDataset(base_path, ensemble_synth_ids, 1900, 1950)
test_ds = PredictSameDataset(base_path, ensemble_synth_ids, 1950, 2000)

train_loader = DataLoader(train_ds, batch_size = 200, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 200, shuffle = True)
test_loader = DataLoader(test_ds, batch_size = 200)

model = CNNModel(12).to(device)
loss_func = nn.CrossEntropyLoss()

train_time = train_model(model, train_loader, val_loader, loss_func, 5)
torch.save(model.state_dict(), model_name)

print("=====")
print("Accuracy", get_accuracy(model, test_loader))
print("=====")