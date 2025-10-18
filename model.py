import torch.nn as nn

class SignalClassifier(nn.Module): # linear layers with relu activations
    def __init__(self):
        super(SignalClassifier, self).__init__()
        self.linear1 = nn.Linear(300, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(64, 3)
    
    def forward(self, x): 
        x = self.linear1(x)
        x = self.relu(x)  
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x