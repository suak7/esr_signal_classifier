# AI Electron Spin Resonance (ESR) Signal Classifier

<h2>Inspiration</h2>
<p>My motivation behind this project sparked a while after my junior year of high school when I worked on electron spin resonance research to study organic semiconductors and DNA nanocrystals. I gained intermediate experience in research techniques, such as freeze pump thaw, X-ray diffraction, electron microscopy, ball milling, and magic angle spinning. I wished for my first personal project to connect back to a learning experience, so I chose my previous lab research experience to make it an informative, enjoyable journey for myself.</p>

<h2>Project Overview</h2>
<p>A PyTorch-based neural network that demonstrates AI signal classification, dataset generation, model training, and performance visualization. The goal is to classify synthetic electron spin resonance (ESR) signals into three distinct classes based on their characteristic patterns:</p>
<ul>
  <li>Class 0: Single Gaussian-like peak</li>
  <li>Class 1: Double Gaussian peaks</li>
  <li>Class 2: Irregular / high-frequency noisy pattern</li>
</ul>

The dataset is generated programmatically using Python and is designed to mimic realistic ESR measurements, including natural variability and noise.

<h2>Features</h2>
<ul>
  <li>Generates synthetic signals with different patterns: Class 0, Class 1, and Class 2</li>
  <li>Train/test split with PyTorch DataLoaders</li>
  <li>Noise addition to simulate realistic experimental conditions</li>
  <li>Splitting of dataset into training and test sets using PyTorch DataLoader</li>
  <li>Saves best model for later inference</li>
  <li>Visualizes performance curves</li>
</ul>

## Getting Started

Requirements:
<ul>
  <li>Python 3.13+</li>
  <li><a href="https://numpy.org/install/">NumPy</a></li>
  <li><a href="https://matplotlib.org/stable/install/index.html">Matplotlib</li>
  <li><a href="https://pytorch.org/get-started/locally/">PyTorch</li>
</ul>