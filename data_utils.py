"""
PyTorch Data Loading Utilities
"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from config import DATASET_CONFIG, MODEL_CONFIG, DEVICE

class CattleDataset(Dataset):
    """Custom PyTorch Dataset for cattle images"""
    
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def find_dataset():
    """Find dataset folder"""
    for path in DATASET_CONFIG['possible_paths']:
        if os.path.exists(path):
            return path
    return None

def load_dataset():
    """Load and preprocess the cattle dataset"""
    dataset_path = find_dataset()
    if not dataset_path:
        print("❌ Dataset not found!")
        return None, None, None
    
    print(f"✅ Found dataset at: {dataset_path}")
    
    # Explore dataset
    breeds = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            image_files = [f for f in os.listdir(item_path) 
                          if f.lower().endswith(DATASET_CONFIG['supported_formats'])]
            breeds.append({'breed': item, 'files': len(image_files)})
    
    breeds = sorted(breeds, key=lambda x: x['files'], reverse=True)
    print(f"📊 Found {len(breeds)} breeds")
    
    # Load images
    images = []
    labels = []
    
    print("📸 Loading images...")
    for breed_info in breeds:
        breed = breed_info['breed']
        breed_path = os.path.join(dataset_path, breed)
        
        image_files = [f for f in os.listdir(breed_path) 
                      if f.lower().endswith(DATASET_CONFIG['supported_formats'])]
        image_files = image_files[:DATASET_CONFIG['max_images_per_breed']]
        
        print(f"Processing {breed}... ", end="")
        count = 0
        
        for img_file in image_files:
            img_path = os.path.join(breed_path, img_file)
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, DATASET_CONFIG['image_size'])
                
                images.append(image)
                labels.append(breed)
                count += 1
            except:
                continue
        
        print(f"Loaded {count} images")
    
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    
    print(f"✅ Dataset loaded: {len(images)} images, {len(label_encoder.classes_)} breeds")
    
    return np.array(images), encoded_labels, label_encoder

def get_data_transforms():
    """Get data augmentation transforms"""
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation/test transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(X_train, y_train, X_val, y_val):
    """Create PyTorch DataLoaders"""
    
    train_transform, val_transform = get_data_transforms()
    
    # Create datasets
    train_dataset = CattleDataset(X_train, y_train, train_transform)
    val_dataset = CattleDataset(X_val, y_val, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=True,
        num_workers=MODEL_CONFIG['num_workers'],
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=MODEL_CONFIG['batch_size'],
        shuffle=False,
        num_workers=MODEL_CONFIG['num_workers'],
        pin_memory=True if DEVICE.type == 'cuda' else False
    )
    
    return train_loader, val_loader
