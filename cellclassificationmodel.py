import os
import subprocess
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import random
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Ensure malaria dataset exists
DATA_DIR = Path('data/malaria/cell_images')
ZIP_PATH = Path('data/malaria/cell-images-for-detecting-malaria.zip')

if not DATA_DIR.exists():
    # Download the zip only if it's not already present
    if not ZIP_PATH.exists():
        ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            'kaggle datasets download -d iarunava/cell-images-for-detecting-malaria -p data/malaria -q',
            shell=True, check=True)
        # shell, inside shell command and check reaises an error if command fails. 
    # Unzip to data/malaria/
    subprocess.run(f'unzip -q -o {ZIP_PATH} -d data/malaria', shell=True, check=True)


#Create train/test split (70/30) 
TRAIN_DIR = Path('data/malaria/train')
TEST_DIR  = Path('data/malaria/test')

if not TRAIN_DIR.exists() and not TEST_DIR.exists():
    random.seed(42)
    for cls in ['Parasitized', 'Uninfected']:
        src_dir = DATA_DIR / cls
        # gets all the png files from the folder and puts them in a list
        # what is alex net and vgg network.
        images  = list(src_dir.glob('*.png'))
        random.shuffle(images)
        k = int(0.30 * len(images))  
        test_imgs  = images[:k]
        train_imgs = images[k:]

        # Create destination folders
        (TEST_DIR  / cls).mkdir(parents=True, exist_ok=True)
        (TRAIN_DIR / cls).mkdir(parents=True, exist_ok=True)

        # Move files
        for p in test_imgs:
            shutil.move(str(p), TEST_DIR  / cls / p.name)
        for p in train_imgs:
            shutil.move(str(p), TRAIN_DIR / cls / p.name)




# Data transforms and DataLoaders
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation for RGB color channels
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    # no augmentation for test 
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets that atuomatically loads images from folders and transfoms them
train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

# Create data loaders, batch of 32 images, shuffling and then no multithreading
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")



# use gpu 
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


import torch.nn as nn
import torch.nn.functional as F

class ParasiteDetector(nn.Module):
    def __init__(self):
        # constructor to initialize the model
        super(ParasiteDetector, self).__init__()
        
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    #  3 rgb channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 32 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1) #  128 channels 
        # each time it takes more and more data and features into consideration
        # vgg arhitecutre, 
        
        # takes the max of a 2*2 region and reduces the image size by 2 and helps the model focus on the strong features. 
        self.pool = nn.MaxPool2d(2, 2)
        
        # Randomly turns off 50% of neurons during training to prevent overfitting
        self.dropout = nn.Dropout(0.5)
        
        
        # After 4 pooling operations: 224/16 = 14, so 14x14x256 = 50176
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 classes: Parasitized, Uninfected
        
    def forward(self, x):
        # Feature extraction with ReLU activation and pooling
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28
        x = self.pool(F.relu(self.conv4(x)))  # 28 -> 14

        """ self.conv1(x) - Applies convolution filters to detect features
        F.relu() - Applies ReLU activation (turns negative values to 0)
        self.pool() - Applies max pooling (reduces size by half) """
        
        # Flatten for fully connected layers
        x = x.view(-1, 256 * 14 * 14)
        # fundamental feed forward neural netwrok classfication (activation function)
        
        # Classification layers 
        x = F.relu(self.fc1(x))
        # randomly turns off 50% of the neurons to prevent overfitting
       
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# Create model instance and move to device
model = ParasiteDetector().to(device)
print(f"Model created and moved to: {device}")

import torch.optim as optim

# Loss function 
criterion = nn.CrossEntropyLoss()
# optimzes learning rate
# check all the possible differrent models
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
num_epochs = 5

print("Starting training...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
     
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        # Zero gradients
        # how to add the error plotting real time to see if loss reduces. 
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        """ Converts raw scores to predictions (0=Parasitized, 1=Uninfected). """
        _,predicted = torch.max(outputs.data, 1)
        # counts the number of images in the batch
        total_train += labels.size(0)
        # counts the number of correct predictions
        correct_train += (predicted == labels).sum().item()
        
        # Print progress every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {loss.item():.4f}')
    
    # Print epoch results 
    train_acc = 100 * correct_train / total_train
    avg_train_loss = running_loss / len(train_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}]:')
    print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print('-' * 60)

print("Training completed!")

# Test
print("Evaluating on test set...")
model.eval()
correct_test = 0
total_test = 0
test_loss = 0.0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        _,predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        
        # Collect for confusion matrix
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Final test results
test_acc = 100 * correct_test / total_test
avg_test_loss = test_loss / len(test_loader)

print(f'Final Test Results:')
print(f'  Test Loss: {avg_test_loss:.4f}')
print(f'  Test Accuracy: {test_acc:.2f}%')
print('=' * 60)


print("Creating confusion matrix...")

class_names = ['Parasitized', 'Uninfected']
cm = confusion_matrix(all_labels, all_predictions)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print("Confusion matrix saved as 'confusion_matrix.png'")


### how to save the model. How to predict for the new image. 

#### what is the difference between deep learning classifcation and feautre vector classification


