import numpy as np
import matplotlib.pyplot as plt
import random

x = np.linspace(0, 1, 100) # 100 points between 0 and 1

def generate_class0_signal(x): # single peak (Gaussian-like)
    center = np.random.uniform(0.4, 0.6)
    width = np.random.uniform(0.08, 0.12)

    signal = np.exp(-((x - center)**2) / (2 * width**2))
    noise = np.random.normal(0, 0.05, size=x.shape)
    signal = signal + noise

    return signal

def generate_class1_signal(x): # double peak 
    center1 = np.random.uniform(0.32, 0.34)
    center2 = np.random.uniform(0.65, 0.66)
    width1 = np.random.uniform(0.07, 0.09)
    width2 = np.random.uniform(0.11, 0.13)

    peak1 = np.exp(-((x - center1)**2) / (2 * width1**2))
    peak2 = np.exp(-((x - center2)**2) / (2 * width2**2))
    signal = peak1 + peak2

    noise = np.random.normal(0, 0.05, size=x.shape)
    signal = signal + noise

    return signal

def generate_class2_signal(x): # irregular peaks with increased noise
    signal = 1.0*(np.sin(5*x) + 0.5*np.sin(8*x))
    noise = np.random.normal(0, 0.07, size=x.shape)
    signal = signal + noise

data = []
labels = []

num_samples_per_class = 300

for class_label in [0, 1, 2]: # generating samples for each class
    for i in range(num_samples_per_class):
        if class_label == 0:
            signal = generate_class0_signal(x)
        elif class_label == 1:
            signal = generate_class1_signal(x)
        else:
            signal = generate_class2_signal(x)
        
        data.append(signal)
        labels.append(class_label)

combined = list(zip(data, labels)) # randomly mixing signals and labels
random.shuffle(combined)
data, labels = zip(*combined)

data = np.array(data)
labels = np.array(labels)

np.save("signals.npy", data)
np.save("labels.npy", labels)