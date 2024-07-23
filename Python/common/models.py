import torch
import torch.nn as nn
import numpy as np
from .config import device


def load_model(synth_path, filename):
    y = np.genfromtxt(synth_path+"/labels/0.csv", delimiter=",", dtype=np.float64, usemask = False)
    O = y.shape[0]-1
    model = CNNModel(O).to(device)
    model.load_state_dict(torch.load(synth_path + filename))    
    model.eval()
    return model

def init_weights(m):
  if type(m) == nn.Linear:
    nn.init.xavier_uniform(m.weight)
    m.bias.data.fill_(0.0)
  if isinstance(m, nn.Conv2d):
    nn.init.xavier_uniform(m.weight)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, 
                            bidirectional = True, batch_first =True)
        self.linear1 = nn.Linear(hidden_size * 2, output_size)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)


    def forward(self, x):
        x = x.transpose(1,2)
        lstm_out, h = self.lstm(x)
        final_hidden = h[0]
        final_hidden = torch.cat([final_hidden[0], final_hidden[1]], 1)
        y = self.sigmoid( self.linear1(final_hidden))
        return  y
    
class CNNModel(nn.Module):
    def __init__(self, output_size):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(1, 64, 11, (1,3)), 
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, 3, (1,3)), 
                nn.LeakyReLU(),
                nn.Conv2d(128, 256, 3, (1,3)), 
                nn.LeakyReLU(),
                nn.Conv2d(256, 128, 3, padding = "same"), 
                nn.LeakyReLU(),
                nn.Conv2d(128, 64, 3, padding = "same"), 
                nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(4992, 4000),
                nn.LeakyReLU(),
                nn.Linear(4000, 3000),
                nn.LeakyReLU(),
                nn.Linear(3000, 2000),
                nn.LeakyReLU(),
                nn.Linear(2000, 1800),
                nn.LeakyReLU(),
                nn.Linear(1800, 1600),
                nn.LeakyReLU(),
                nn.Linear(1600, 1400),
                nn.LeakyReLU(),
                nn.Linear(1400, 1200),
                nn.LeakyReLU(),
                nn.Linear(1200, 1000),
                nn.LeakyReLU(),
                nn.Linear(1000, 800),
                nn.LeakyReLU(),
                nn.Linear(800, 600),
                nn.LeakyReLU(),
                nn.Linear(600, 400),
                nn.LeakyReLU(),
                nn.Linear(400, output_size),
                nn.LeakyReLU(),
        )
        self.apply(init_weights)

    def forward(self, x):
            x1 = x[:,None,:,:]
            return self.model(x1)    
    
  
        
class MLPModel(nn.Module):        
        def __init__(self, input_size, output_size):
            super(MLPModel, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_size, 5000),
                nn.LeakyReLU(),
                nn.Linear(5000, 4000),
                nn.LeakyReLU(),
                nn.Linear(4000, 3000),
                nn.LeakyReLU(),
                nn.Linear(3000, 2000),
                nn.LeakyReLU(),
                nn.Linear(2000, 1800),
                nn.LeakyReLU(),
                nn.Linear(1800, 1600),
                nn.LeakyReLU(),
                nn.Linear(1600, 1400),
                nn.LeakyReLU(),
                nn.Linear(1400, 1200),
                nn.LeakyReLU(),
                nn.Linear(1200, 1000),
                nn.LeakyReLU(),
                nn.Linear(1000, 800),
                nn.LeakyReLU(),
                nn.Linear(800, 600),
                nn.LeakyReLU(),
                nn.Linear(600, 400),
                nn.LeakyReLU(),
                nn.Linear(400, output_size),
                nn.LeakyReLU(),
            )
            self.apply(init_weights)
            
        def forward(self, x):
            x = x.flatten(1)
            return self.model(x)
    
