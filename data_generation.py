import numpy as np

x = np.linspace(0, 1, 1000) # 1000 points between 0 and 1

def generate_class0_signal(x): # single peak (Gaussian-like)
    center = np.random.uniform(0.35, 0.65)
    width = np.random.uniform(0.07, 0.15)
    amplitude = np.random.uniform(0.8, 1.2)
    signal = amplitude * np.exp(-((x - center)**2) / (2 * width**2))
    noise = np.random.normal(0, 0.1, size=x.shape)  # increased noise
    return signal + noise

def generate_class1_signal(x): # double peak 
    center1 = np.random.uniform(0.25, 0.40)
    center2 = np.random.uniform(0.60, 0.75)
    width1 = np.random.uniform(0.07, 0.15)
    width2 = np.random.uniform(0.07, 0.15)
    amplitude1 = np.random.uniform(0.8, 1.2)
    amplitude2 = np.random.uniform(0.8, 1.2)

    peak1 = amplitude1 * np.exp(-((x - center1)**2) / (2 * width1**2))
    peak2 = amplitude2 * np.exp(-((x - center2)**2) / (2 * width2**2))
    signal = peak1 + peak2

    noise = np.random.normal(0, 0.1, size=x.shape)
    return signal + noise

def generate_class2_signal(x): # irregular peaks with increased noise
    scale1 = np.random.uniform(0.8, 1.2)
    scale2 = np.random.uniform(0.4, 0.6)
    signal = scale1 * np.sin(5*x) + scale2 * np.sin(8*x)
    noise = np.random.normal(0, 0.15, size=x.shape)  # higher noise
    return signal + noise

data = []
labels = []

num_samples_per_class = 1000

for class_label in [0, 1, 2]: # generating samples for each class
    for i in range(num_samples_per_class):
        if class_label == 0:
            signal = generate_class0_signal(x)
        elif class_label == 1:
            signal = generate_class1_signal(x)
        else:
            signal = generate_class2_signal(x)

        signal = np.array(signal, dtype=np.float32)
        
        data.append(signal)
        labels.append(class_label)

data = np.stack(data)               
labels = np.array(labels, dtype=np.int64)
indices = np.arange(len(data))
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
  
np.save("signals.npy", data)
np.save("labels.npy", labels)