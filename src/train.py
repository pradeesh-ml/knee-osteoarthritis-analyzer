# =================== Imports ===================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils
from torchsummary import summary

# =================== Paths ===================
BASE_DIR = os.getcwd()
train_path = os.path.join(BASE_DIR,'knee_osteoarthritis_dataset', 'train')
val_path = os.path.join(BASE_DIR,'knee_osteoarthritis_dataset', 'val')
test_path = os.path.join(BASE_DIR,'knee_osteoarthritis_dataset', 'test')

# =================== Data Loader ===================
def load_data(name, path):
    image_paths, labels = [], []
    classes = sorted(os.listdir(path))

    for class_name in classes:
        class_dir = os.path.join(path, class_name)
        for img in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img)
            if os.path.isfile(img_path):
                image_paths.append(img_path)
                labels.append(class_name)

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    label_indices = [class_to_idx[label] for label in labels]

    data = pd.DataFrame({
        'Image_path': image_paths,
        'Label': label_indices
    })

    print(f"{name} Distribution:", data['Label'].value_counts().to_dict(), "\n")
    return data

train_data = load_data('Train', train_path)
val_data = load_data('Validation', val_path)
test_data = load_data('Test', test_path)

# =================== Transforms ===================
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# =================== Custom Dataset ===================
class KneeDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'Image_path']
        label = int(self.df.loc[idx, 'Label'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

# =================== Loaders ===================
train_dataset = KneeDataset(train_data, train_transform)
val_dataset = KneeDataset(val_data, test_transform)
test_dataset = KneeDataset(test_data, test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# =================== Visualize Sample ===================
def visualize_image(img_tensor):
    img = img_tensor.numpy().transpose((1, 2, 0))
    img = img * std + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.axis('off')

images, _ = next(iter(train_loader))
plt.figure(figsize=(10, 10))
visualize_image(utils.make_grid(images))
plt.show()


# -------------------- Model Setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 5)
model = model.to(device)

summary(model, (3, 224, 224))

# -------------------- Class Weights --------------------
all_labels = train_data['Label'].values
class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# -------------------- Loss, Optimizer --------------------
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------- Combine all data for final training --------------------
combined_data = pd.concat([train_data, val_data, test_data], ignore_index=True).sample(frac=1).reset_index(drop=True)
combined_dataset = KneeDataset(combined_data, transform=train_transform)
train_loader = DataLoader(combined_dataset, batch_size=32, shuffle=True)

# -------------------- Training Loop --------------------
NUM_EPOCHS = 50
PATIENCE = 15
best_val_loss = float('inf')
early_stop_counter = 0
model_path = 'best_model.pth'

# For plotting
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()

    val_loss /= len(test_loader)
    val_acc = 100 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), model_path)
        print("Validation loss improved. Model saved.")
    else:
        early_stop_counter += 1
        print(f"No improvement. Early stop counter: {early_stop_counter}/{PATIENCE}")
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# -------------------- Plotting --------------------
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label="Train Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy Curve")
plt.legend()

plt.savefig('graph.png')
plt.tight_layout()
plt.show()