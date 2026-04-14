# Project 2: Image Classification with Deep Learning (CNN) 🖼️

This project demonstrates how to build and train a Convolutional Neural Network (CNN) using **PyTorch**, the industry standard for Deep Learning and Computer Vision. 

## The Goal
To build an AI model that can look at an image and correctly classify what it sees. In this beginner-friendly example, we will train our CNN to distinguish between different geometric shapes (Circles vs. Squares). 

## Skills Demonstrated
*   **Deep Learning Framework:** `PyTorch` (`torch`, `torchvision`)
*   **Computer Vision:** Convolutional Neural Networks (CNNs), Image transformations, data loaders.
*   **Model Architecture:** Convolutional layers, Max Pooling, Fully Connected (Linear) layers, Activation functions (ReLU).
*   **Training Loop:** Forward pass, Loss calculation (CrossEntropy), Backpropagation, Optimizer steps (Adam).

## 📁 Files Explained

1.  **`dataset_generator.py`**: Instead of having you download a massive gigabyte-sized dataset, this script procedurally generates thousands of images of circles and squares and organizes them into the correct `train` and `test` folder structure that PyTorch expects.
2.  **`train_cnn.py`**: The core deep learning script. It defines the neural network architecture, loads the image data, trains the model over multiple "epochs," and tests its accuracy.

## 🚀 How to Run (Beginner Friendly)

**Step 1:** Install PyTorch and Pillow (for image processing). 
Run this in your terminal:
```powershell
py -m pip install torch torchvision torchaudio pillow matplotlib
```

**Step 2:** Generate the Image Dataset:
```powershell
py dataset_generator.py
```
*(This will create a `data/` folder filled with generated images).*

**Step 3:** Train the Deep Learning Model:
```powershell
py train_cnn.py
```

## Real-World Intern Use Cases 💡
*   **Manufacturing:** Using cameras to classify products on an assembly line as "defective" or "normal".
*   **Healthcare:** Classifying X-ray scans into "pneumonia" vs "healthy".
*   **Retail:** Automatically tagging and categorizing clothing images uploaded by users.