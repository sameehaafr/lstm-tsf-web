import torch
import torch.nn as nn
import h5py
import numpy as np

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def convert_h5_to_pt():
    # Create model instance
    model = LSTMModel(input_size=1, hidden_size=50, num_layers=1, dropout=0.6)
    
    # Load weights from HDF5 file
    with h5py.File('models/lstm_model_10.h5', 'r') as f:
        # Load the weights from the HDF5 file
        for key in f.keys():
            if key in model.state_dict():
                model.state_dict()[key].copy_(torch.from_numpy(f[key][:]))
    
    # Save in PyTorch format
    torch.save(model.state_dict(), 'models/lstm_model_10.pt')
    print("Model converted and saved successfully!")

if __name__ == "__main__":
    convert_h5_to_pt() 