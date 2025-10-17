# AI ESR Signal Classifier

<h2>Project Overview</h2>
<p>This project implements an AI-based signal classifier for ESR-like data. The goal is to classify synthetic electron spin resonance (ESR) signals into three distinct classes based on their characteristic patterns:</p>
<ul>
  <li>Class 0: Single Gaussian-like peak</li>
  <li>Class 1: Double Gaussian peaks</li>
  <li>Class 2: Irregular / high-frequency noisy pattern</li>
</ul>

The dataset is generated programmatically using Python and is designed to mimic realistic ESR measurements, including natural variability and noise. This project serves as a proof-of-concept for AI signal classification using PyTorch and demonstrates a combination of signal processing, machine learning, and scientific computing.

<h2>Features</h2>
<ul>
  <li>Synthetic generation of ESR-like signals with controlled variability.</li>
  <li>Modular functions for generating each signal class.</li>
  <li>Noise addition to simulate realistic experimental conditions.</li>
  <li>Splitting of dataset into training and test sets using PyTorch DataLoader.</li>
  <li>Fully compatible with PyTorch for building 1D signal classification models.</li>
</ul>

<h2>Getting Started</h2>
Requirements:
<ul>
  <li>Python 3.8+</li>
  <li>NumPy</li>
  <li>Matplotlib</li>
  <li>PyTorch</li>
</ul>
Install required packages: 
<ul>
  <li><a href="https://numpy.org/install/">NumPy</a></li>
  <li><a href="https://matplotlib.org/stable/install/index.html">Matplotlib</li>
  <li><a href="https://pytorch.org/get-started/locally/">PyTorch</li>
</ul>

<h3>Generate Dataset</h3>
<p>This will generate signals.npy and labels.npy containing 300 samples per class by default.</p>
python data_generation.py 

<h3>Load Dataset and Create Train/Test Sets</h3>
<p>This script converts the signals and labels into PyTorch tensors and splits them into training and test datasets using DataLoader.</p>
python dataset.py 

<h2>Usage</h2>
<p>Once the dataset is ready, you can train a PyTorch model (like a simple 1D CNN or fully connected network) to classify the signals.</p>

1. Generate dataset (data_generation.py)
2. Load dataset and create DataLoaders (dataset.py)
3. Build and train your model (model.py)
4. Evaluate model accuracy on test set
