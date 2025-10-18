import torch
import torch.nn as nn
class SignalClassifier(nn.Module): # linear layers with dropouts
    def __init__(self):
        super(SignalClassifier, self).__init__()
        self.linear1 = nn.Linear(1000, 256)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)  
        self.linear2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.2)  
        self.linear3 = nn.Linear(128, 3)
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        return x