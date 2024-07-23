import numpy as np
import torch
import torch.nn as nn
import time
from .config import device

def train_model(model, train_loader, val_loader, 
    loss_func = nn.MSELoss(),
    patience = 10, num_epochs = 100
    ):

    counter = 0
    min_val_loss = None
    best_model = None

    loss_func=nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)#, weight_decay=0.001

    loss_curve = []
    val_loss_curve = []
    start = time.time()
    for n in range(num_epochs):
        loss_acc = 0
        i = 0
        for x_batch, y_batch in train_loader:
            y_pred = model(x_batch.to(device))
            loss = loss_func(y_pred, y_batch.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_acc+=loss
            i= i+1
        print("---- loss ",n, loss_acc.item() / i)
        loss_curve.append(loss_acc.item() / i)
    
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                vy_pred = model(val_x.to(device))
                tmp_loss = loss_func(vy_pred, val_y.to(device))
                val_loss += tmp_loss
            val_loss_curve.append(val_loss.item() / len(val_loader))
    
        print("---- val loss ",n, val_loss.item())
        if(min_val_loss is None or  val_loss < min_val_loss):
            min_val_loss = val_loss
            counter = 0
            best_model = model.state_dict()
        elif val_loss > min_val_loss:
            counter = counter + 1
            if counter >= patience:
                print("stopping", n)
                break
    model.load_state_dict(best_model)
    print("=====")
    print("Train time ",time.time() - start)
    print("Val loss", val_loss.item())
    train_time = time.time() - start
    return train_time