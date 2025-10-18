import numpy as np
import torch
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader

data = np.load("signals.npy")
labels = np.load("labels.npy")
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(data_tensor, labels_tensor)

num_samples = len(dataset)
train_size = int(0.8 * num_samples) # split is 80% training and 20% test
test_size = num_samples - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) # random split

# batches of data to feed the model
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)