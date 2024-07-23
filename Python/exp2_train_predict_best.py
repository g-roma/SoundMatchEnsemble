#!/usr/bin/env python3
# train "predict best" classifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from common.models import CNNModel
from common.config import device, ensemble_synth_ids, nsynth_train
from common.datasets import PredictBestDataset
from common.train import train_model
from common.evaluation import get_accuracy

model_name = "exp2_predict_best.pt"
ds = PredictBestDataset(nsynth_train, "predict_best_dataset.csv", ensemble_synth_ids)

train_ds = Subset(ds, range(0, 3800))
val_ds = Subset(ds, range(3800, 3900))
test_ds = Subset(ds, range(3900, 4000))

train_loader = DataLoader(train_ds, batch_size = 200, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = 200, shuffle = True)
test_loader = DataLoader(test_ds, batch_size = 200)

model = CNNModel(12).to(device)
loss_func = nn.CrossEntropyLoss()

train_model(model, train_loader, val_loader, loss_func, 5)
torch.save(model.state_dict(), model_name)
print("=====")
print("Accuracy", get_accuracy(model, test_loader))
print("=====")