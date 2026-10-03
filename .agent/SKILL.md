# Domain Knowledge: Audio Noise Suppression via Computer Vision

This document outlines the core concepts, architectural decisions, and machine learning techniques utilized in this non-stationary noise suppression project. It serves as a knowledge base for understanding the intersection of Digital Signal Processing (DSP) and Deep Learning within the constraints of Edge Computing (TinyML).

## 1. Digital Signal Processing (DSP) Fundamentals
* **STFT (Short-Time Fourier Transform):** Converts 1D time-domain audio signals into 2D frequency-domain representations (Spectrograms). This allows the neural network to process audio using Computer Vision techniques.
* **iSTFT (Inverse STFT):** Reconstructs the filtered 1D time-domain audio signal from the masked magnitude spectrogram and the original noisy phase.
* **Ideal Ratio Mask (IRM):** The ground-truth target for the neural network. It is calculated as the ratio of clean speech power to the sum of clean speech and noise power: `IRM = (|S|^2 / (|S|^2 + |N|^2))^beta`. Its values strictly range from 0.0 to 1.0.

## 2. Machine Learning Engineering & Preprocessing
* **Logarithmic Compression:** Raw STFT magnitude has an massive, asymmetric dynamic range. Applying logarithmic compression (e.g., `np.log1p()`) stabilizes the data, preventing exploding gradients and allowing the network to learn morphological patterns rather than just lowering overall volume.
* **Normalization (Z-Score):** Shifting the compressed spectrogram to have a mean of 0 and a standard deviation of 1 ensures stable and faster convergence during training.
* **Data Leakage Prevention:** When splitting datasets (Train/Validation), splits must be done based on the *source clean audio files*, not on the generated noisy samples. This prevents the model from memorizing specific speakers (overfitting) and ensures true generalization.
* **I/O Optimization (Precomputing):** Performing STFT and matrix additions on-the-fly inside the PyTorch `DataLoader` creates severe CPU bottlenecks. Precomputing these tensors and saving them as `.pt` files accelerates training by orders of magnitude.

## 3. Deep Learning Architecture (TinyU-Net)
* **Spatial Filtering:** Instead of generating audio directly (which is computationally expensive), the U-Net acts as a spatial filter. It predicts a 2D mask (gain matrix) that is element-wise multiplied by the noisy input spectrogram.
* **TinyML Constraints:** The architecture is heavily bottlenecked (e.g., starting with 8 channels) to minimize Multiply-Accumulate (MAC) operations and memory footprint, making it viable for future microcontroller deployment.
* **Skip Connections:** Essential for preserving high-frequency spatial details that are lost during the Encoder's Max-Pooling operations.
* **Activation & Loss:** The final layer uses a Sigmoid activation function to force outputs between 0 and 1 (matching the IRM), optimized using Binary Cross-Entropy (BCELoss).
* **Learning Rate Scheduling:** Utilizing techniques like `ReduceLROnPlateau` prevents the validation loss from bouncing out of local minima, ensuring smooth convergence.