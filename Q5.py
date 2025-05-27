import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ca_utils import ResNet, BasicBlock

def test_cnn():
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transformations for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform_test)
    
    # Define the animal classes in CIFAR-10
    animal_classes = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']
    animal_indices = [2, 3, 4, 5, 6, 7]  # Corresponding indices in CIFAR-10
    
    # Filter the test set to include only animal classes
    test_indices = [i for i, (_, label) in enumerate(testset) if label in animal_indices]
    animal_testset = torch.utils.data.Subset(testset, test_indices)
    
    # Create test data loader
    testloader = torch.utils.data.DataLoader(animal_testset, batch_size=100, shuffle=False)
    
    # Initialize the model with the same architecture used in training
    model = ResNet(block=BasicBlock, layers=[1, 1, 1], num_classes=6)
    
    # Load the trained model weights
    model.load_state_dict(torch.load('data/weights_resnet.pth', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Prepare to collect predictions and true labels
    all_predictions = []
    all_true_labels = []
    correct = 0
    total = 0
    
    # Test the model
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            
            # Convert original CIFAR labels to our 0-5 range for animal classes
            remapped_labels = torch.tensor([animal_indices.index(label.item()) for label in labels])
            all_true_labels.extend(remapped_labels.numpy())
            
            # Get model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            
            # Calculate accuracy
            total += remapped_labels.size(0)
            correct += (predicted.cpu() == remapped_labels).sum().item()
    
    # Convert to numpy arrays with int64 data type as required
    all_predictions = np.array(all_predictions, dtype=np.int64)
    all_true_labels = np.array(all_true_labels, dtype=np.int64)
    
    # Calculate overall accuracy
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    
    # Generate and display confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=animal_classes, yticklabels=animal_classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    return all_predictions, accuracy

if __name__ == "__main__":
    # Test the CNN
    predictions, accuracy = test_cnn()
    
    # Save the predictions
    np.save('test_predictions.npy', predictions)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Number of predictions: {len(predictions)}")
    print("Testing completed successfully.")