import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ---------------------------------------------
# 1. DEFINE THE NEURAL NETWORK (THE AI BRAIN)
# ---------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        """
        A beginner-friendly Convolutional Neural Network.
        Here we define the layers of the network.
        CNNs learn visual patterns: edges, textures, shapes.
        """
        super(SimpleCNN, self).__init__()
        
        # Layer 1: Learn low-level features (e.g., edges)
        # Input channels=3 (RGB), Output channels=16 filters, kernel_size=3x3 window
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU() # Activation function: adds non-linearity
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Downsample the image
        
        # Layer 2: Learn high-level features (e.g., shapes)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Layer 3: Fully Connected (Linear) layer: Makes the final classification decision
        # Image was 64x64, pooled twice (64->32->16). So size is 16x16.
        # Channels is 32. Total flattened size = 32 * 16 * 16 = 8192
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu3 = nn.ReLU()
        
        # Final Layer: Output layer for the two classes
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        This defines how the image data flows through the layers.
        """
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Flatten the 3D tensor into a 1D array to pass to the linear layer
        x = x.view(-1, 32 * 16 * 16)
        
        x = self.relu3(self.fc1(x))
        x = self.fc2(x) # Output the raw class scores (logits)
        return x

# ---------------------------------------------
# 2. DATA PREPARATION (LOADING IMAGES)
# ---------------------------------------------
def load_data(data_dir):
    """
    Load images, resize them, and convert them to PyTorch Tensors (tensors are 3D numbers).
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    
    if not os.path.exists(train_dir):
        print("Error: Dataset not found. Please run 'python dataset_generator.py' first!")
        return None, None
    
    # Transformations applied to the images
    transform = transforms.Compose([
        transforms.Resize((64, 64)),   # Ensure all images are exactly 64x64
        transforms.ToTensor(),         # Convert image to numerical format between 0-1
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize RGB channels Let mean=0
    ])

    # PyTorch's ImageFolder automatically uses subfolders as class names ('circle', 'square')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # DataLoaders fetch the images in batches (so we don't load 10,000 images into memory at once)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Get the class names (should be ['circle', 'square'])
    classes = train_dataset.classes
    print(f"Classes found: {classes}")
    
    return train_loader, test_loader

# ---------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------
def train_model():
    print("====================================")
    print("1. SETTING UP DEVICE (CPU/GPU)")
    print("====================================")
    # PyTorch can run on the GPU for massive speedups (if you have one)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    train_loader, test_loader = load_data(data_dir)
    if not train_loader: return
    
    # Initialize the Neural Network and move it to the device
    model = SimpleCNN(num_classes=2).to(device)
    
    # The Loss Function (How wrong is the model?) - CrossEntropy is used for classification
    criterion = nn.CrossEntropyLoss()
    
    # The Optimizer (How do we update the numbers to fix the errors?) - Adam is very popular
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 3 # How many times we loop over the entire dataset
    
    print("\n====================================")
    print("2. STARTING TRAINING (epochs = 3)")
    print("====================================")
    
    # Loop over the dataset multiple times
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        # Loop over batches of 32 images at a time
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Step 1: Zero the parameter gradients so they don't accumulate
            optimizer.zero_grad()
            
            # Step 2: Forward pass (Ask the model to predict the classes)
            outputs = model(images)
            
            # Step 3: Compute Loss (Compare predictions vs actual answers)
            loss = criterion(outputs, labels)
            
            # Step 4: Backward pass (Calculate the gradients - backpropagation)
            loss.backward()
            
            # Step 5: Optimize (Update the weights based on gradients)
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {running_loss/len(train_loader):.4f}")

    print("Training Complete!")
    
    print("\n====================================")
    print("3. EVALUATING ACCURACY ON TEST DATA")
    print("====================================")
    # Testing mode: We don't want to update weights anymore, just check performance.
    model.eval()
    correct = 0
    total = 0
    
    # with torch.no_grad() disables backpropagation calculations, saving memory/speed
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Get predictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1) # Get the index of the highest probability
            
            total += labels.size(0) # Count total labels
            correct += (predicted == labels).sum().item() # Count correct predictions

    accuracy = 100 * correct / total
    print(f"Accuracy of the CNN on the test images: {accuracy:.2f}%")
    
    # Save the model state dict (the learned weights)
    model_path = os.path.join(script_dir, 'cnn_model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n=> Hand-trained Model saved to {model_path}!")

if __name__ == '__main__':
    train_model()
