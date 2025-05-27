import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
import xml.etree.ElementTree as ET
import cv2
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
print("Warning: SSL certificate verification disabled")

# MaskedFaceTestDataset class from the problem
class MaskedFaceTestDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        super(MaskedFaceTestDataset, self).__init__()
        self.imgs = sorted(glob.glob(os.path.join(root, '*.png')))
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.imgs)
    
def get_ground_truth_counts(dataset):
   
    print("Extracting ground truth counts from XML files...")
    true_counts = np.zeros((len(dataset), 3), dtype=np.int64)
    
    for idx in range(len(dataset)):
        # Get image path and construct corresponding XML path
        img_path = dataset.imgs[idx]
        xml_path = img_path.replace(".png", ".xml")
        
        # Initialize counts for this image
        with_mask_count = 0
        without_mask_count = 0
        incorrect_mask_count = 0
        
        # Check if XML file exists
        if os.path.exists(xml_path):
            try:
                # Parse XML file
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Count objects by class
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    if class_name == "with_mask":
                        with_mask_count += 1
                    elif class_name == "without_mask":
                        without_mask_count += 1
                    elif class_name == "mask_weared_incorrect":
                        incorrect_mask_count += 1
            except Exception as e:
                print(f"Warning: Could not parse {xml_path}: {e}")
        
        # Store counts for this image
        true_counts[idx] = [with_mask_count, without_mask_count, incorrect_mask_count]
    
    return true_counts

# Implement the mask counting function
def count_masks(dataset):
   
    print("Starting count_masks function")
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Class mapping
    classes = ["background", "with_mask", "without_mask", "mask_weared_incorrect"]
    print("Loading model...")
    
    # Load the pretrained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    
    # Replace the classifier with a new one for our task (4 classes: background + 3 mask classes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
    
    # Load the trained weights
    model.load_state_dict(torch.load('data/weights_counting.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Define transformation for test images
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Prepare arrays to store results
    # Each row will contain counts for [with_mask, without_mask, mask_weared_incorrect]
    all_counts = np.zeros((len(dataset), 3), dtype=np.int64)
    
    # Simulate ground truth counts
    true_counts = np.zeros((len(dataset), 3), dtype=np.int64)
    
    # Process each image
    for idx in range(len(dataset)):
        # Get the image
        image = dataset[idx]

        # Handle image based on its type
        if isinstance(image, torch.Tensor):
            image_tensor = image.to(device)
        else:
            image_tensor = transform(image).to(device)

        # Get model predictions
        with torch.no_grad():
            prediction = model([image_tensor])
        
        # Extract boxes, scores, and labels
        boxes = prediction[0]['boxes'].cpu().numpy()
        scores = prediction[0]['scores'].cpu().numpy()
        labels = prediction[0]['labels'].cpu().numpy()
        
        # Filter detections with confidence > 0.5
        keep = scores > 0.5
        boxes = boxes[keep]
        labels = labels[keep]
        
        # Count instances of each class
        with_mask_count = np.sum(labels == 1)  # with_mask
        without_mask_count = np.sum(labels == 2)  # without_mask
        incorrect_mask_count = np.sum(labels == 3)  # mask_weared_incorrect
        
        # Store counts
        all_counts[idx] = [with_mask_count, without_mask_count, incorrect_mask_count]
        
      
    # Get ground truth counts from XML files
    true_counts = get_ground_truth_counts(dataset)
    
    # Calculate MAPE for each image
    mape_values = []
    for i in range(len(dataset)):
        absolute_percentage_errors = []
        for j in range(3):  # 3 classes
            true_val = true_counts[i, j]
            pred_val = all_counts[i, j]
            
            # Avoid division by zero by using max(true_val, 1)
            denominator = max(true_val, 1)
            ape = abs(true_val - pred_val) / denominator * 100
            absolute_percentage_errors.append(ape)
        
        # Average APE for this image
        mape_values.append(np.mean(absolute_percentage_errors))
    
    # Calculate overall MAPE
    mape = np.mean(mape_values)
    print(f"MAPE: {mape:.2f}%")
    
    return all_counts, mape

def train_mask_detection_model():
    
    print("Starting model training...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define dataset class for training (this handles XML parsing)
    class MaskedFaceTrainDataset(torch.utils.data.Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.imgs = sorted(glob.glob(os.path.join(root, "*.png")))
            self.annotations = [img_path.replace(".png", ".xml") for img_path in self.imgs]
            
            # Class mapping
            self.class_dict = {
                "with_mask": 1,
                "without_mask": 2,
                "mask_weared_incorrect": 3
            }
        
        def __getitem__(self, idx):
            # Load image
            img_path = self.imgs[idx]
            img = Image.open(img_path).convert("RGB")
            
            # Parse XML annotation
            ann_path = self.annotations[idx]
            boxes = []
            labels = []
            
            if os.path.exists(ann_path):
                tree = ET.parse(ann_path)
                root = tree.getroot()
                
                for obj in root.findall("object"):
                    class_name = obj.find("name").text
                    if class_name in self.class_dict:
                        class_id = self.class_dict[class_name]
                        
                        bbox = obj.find("bndbox")
                        xmin = float(bbox.find("xmin").text)
                        ymin = float(bbox.find("ymin").text)
                        xmax = float(bbox.find("xmax").text)
                        ymax = float(bbox.find("ymax").text)
                        
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(class_id)
            
            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            target = {
                "boxes": boxes,
                "labels": labels,
                "image_id": torch.tensor([idx]),
                "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64)
            }
            
            if self.transform is not None:
                img = self.transform(img)
            
            return img, target
        
        def __len__(self):
            return len(self.imgs)
    
    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataset and dataloader
    train_dataset = MaskedFaceTrainDataset("./MaskedFace/train", transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True, 
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Initialize model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)  # background + 3 classes
    model.to(device)
    
    # Define optimizer and parameters
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Training loop 
    num_epochs = 5  # For demonstration - increase for better results
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for images, targets in train_loader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Save model
    os.makedirs('data', exist_ok=True)
    torch.save(model.state_dict(), 'data/weights_counting.pth')
    print("Model saved to data/weights_counting.pth")

# Add this function call to the main section
if __name__ == "__main__":
    # Check if model weights exist, if not train the model
    if not os.path.exists('data/weights_counting.pth'):
        print("Training mask detection model...")
        train_mask_detection_model()
    
    # Rest of your code
    from torchvision import transforms
    test_dataset = MaskedFaceTestDataset(root="./MaskedFace/val", transform=transforms.ToTensor())
    
    # Call function
    counts, mape = count_masks(test_dataset)
    print("Counts:", counts)
    print("MAPE:", mape)