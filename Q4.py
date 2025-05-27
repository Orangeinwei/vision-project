import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from ca_utils import ResNet, BasicBlock

import ssl
ssl._create_default_https_context = ssl._create_unverified_context
print("Warning: SSL certificate verification disabled")

def train_cnn():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations with data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Simpler transformations for validation
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 dataset
    cifar_trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform_train)
    
    # Define the animal classes in CIFAR-10
    animal_classes = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    animal_class_indices = [2, 3, 4, 5, 6, 7]  # Corresponding indices in CIFAR-10
    
    # Find images with animal classes
    animal_image_indices = []
    for i, (_, label) in enumerate(cifar_trainset):
        if label in animal_class_indices:
            animal_image_indices.append(i)
    
    # Split into training (80%) and validation (20%) sets
    np.random.seed(42)
    np.random.shuffle(animal_image_indices)
    split_idx = int(len(animal_image_indices) * 0.8)
    train_indices = animal_image_indices[:split_idx]
    val_indices = animal_image_indices[split_idx:]
    
    # Create samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    # Create data loaders
    trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=64,
                                             sampler=train_sampler, num_workers=2)
    valloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=64,
                                           sampler=val_sampler, num_workers=2)
    
    # Initialize ResNet model with specified configuration
    model = ResNet(block=BasicBlock, layers=[1, 1, 1], num_classes=6)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    # Training parameters
    num_epochs = 50
    best_val_accuracy = 0.0
    
    # Create directory for saving model
    os.makedirs('data', exist_ok=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in trainloader:
            # Filter to only include animal classes
            animal_mask = torch.tensor([label.item() in animal_class_indices for label in labels])
            if not animal_mask.any():
                continue
                
            inputs = inputs[animal_mask].to(device)
            
            # Convert original CIFAR labels (2,3,4,5,6,7) to our 0-5 range for animal classes
            remapped_labels = []
            for label in labels[animal_mask]:
                class_idx = animal_class_indices.index(label.item())
                remapped_labels.append(class_idx)
            
            remapped_labels = torch.tensor(remapped_labels).to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, remapped_labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += remapped_labels.size(0)
            correct += (predicted == remapped_labels).sum().item()
        
        # Print training statistics
        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(trainloader):.4f}, Accuracy: {train_accuracy:.2f}%')
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in valloader:
                # Filter to only include animal classes
                animal_mask = torch.tensor([label.item() in animal_class_indices for label in labels])
                if not animal_mask.any():
                    continue
                    
                inputs = inputs[animal_mask].to(device)
                
                # Convert original CIFAR labels (2,3,4,5,6,7) to our 0-5 range for animal classes
                remapped_labels = []
                for label in labels[animal_mask]:
                    class_idx = animal_class_indices.index(label.item())
                    remapped_labels.append(class_idx)
                
                remapped_labels = torch.tensor(remapped_labels).to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += remapped_labels.size(0)
                val_correct += (predicted == remapped_labels).sum().item()
        
        # Calculate validation accuracy
        val_accuracy = 100 * val_correct / val_total
        print(f'Validation Accuracy: {val_accuracy:.2f}%')
        
        # Save model if it's the best so far
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'data/weights_resnet.pth')
            print(f'Model saved with validation accuracy: {val_accuracy:.2f}%')
        
        # Update learning rate
        scheduler.step()
    
    print(f'Finished Training. Best validation accuracy: {best_val_accuracy:.2f}%')

if __name__ == "__main__":
    # Create output directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Train the CNN
    train_cnn()
    
    print("Training completed successfully.")