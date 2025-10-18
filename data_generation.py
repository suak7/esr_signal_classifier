import numpy as np

x = np.linspace(0, 1, 1000) 

def class0_signal(x): # single peak (Gaussian-like)
    peak_position = np.random.uniform(0.35, 0.65)
    peak_width = np.random.uniform(0.07, 0.15)
    amplitude = np.random.uniform(0.8, 1.2)
    signal = amplitude * np.exp(-((x - peak_position)**2) / (2 * peak_width**2))
    noise = np.random.normal(0, 0.1, size=x.shape)  # increased noise
    return signal + noise

def class1_signal(x): # double peak 
    peak1_position = np.random.uniform(0.25, 0.40)
    peak2_position = np.random.uniform(0.60, 0.75)
    peak1_width = np.random.uniform(0.07, 0.15)
    peak2_width = np.random.uniform(0.07, 0.15)
    amplitude1 = np.random.uniform(0.8, 1.2)
    amplitude2 = np.random.uniform(0.8, 1.2)

    peak1 = amplitude1 * np.exp(-((x - peak1_position)**2) / (2 * peak1_width**2))
    peak2 = amplitude2 * np.exp(-((x - peak2_position)**2) / (2 * peak2_width**2))
    signal = peak1 + peak2

    noise = np.random.normal(0, 0.1, size=x.shape)
    return signal + noise

def class2_signal(x): # irregular peaks with increased noise
    low_freq_amp = np.random.uniform(0.8, 1.2)
    high_freq_amp = np.random.uniform(0.4, 0.6)
    signal = low_freq_amp * np.sin(5*x) + high_freq_amp * np.sin(8*x)
    noise = np.random.normal(0, 0.15, size=x.shape)  # higher noise
    return signal + noise

data = []
labels = []

num_samples_per_class = 1000

for class_label in [0, 1, 2]: # generating samples for each class
    for i in range(num_samples_per_class):
        if class_label == 0:
            signal = class0_signal(x)
        elif class_label == 1:
            signal = class1_signal(x)
        else:
            signal = class2_signal(x)

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