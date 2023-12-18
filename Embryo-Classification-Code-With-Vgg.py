import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import os
from PIL import Image
from PIL import ImageFile
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Chemin vers le répertoire racine de votre jeu de données
data_dir = './data-set-embr/'  

# Transformation pour redimensionner et normaliser les images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Chargement des données d'entraînement, de validation et de test avec des transformations
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

# Création d'un dictionnaire pour mapper les classes et les catégories de qualité
class_quality_mapping = {
    "Rien": 0,
    "t2": 1,
    "tB": 2,
    "tPB2": 3,
    "tPNa": 4
}

# Création des DataLoader pour itérer sur les ensembles de données
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Utilisation d'un modèle pré-entraîné VGG (ou tout autre modèle souhaité)
model = models.vgg16(pretrained=True)

# Remplacement de la dernière couche pour la classification en fonction du nombre total de catégories de qualité
num_ftrs = model.classifier[6].in_features
num_quality_categories = 5

model.classifier[6] = nn.Linear(num_ftrs, num_quality_categories)
print(model)
criterion = nn.CrossEntropyLoss()                        # Utilisation de la perte de l'entropie croisée pour la classification
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Entraînement du modèle
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
num_epochs = 15
best_val_acc = 0.0

# Initialisation des listes pour les courbes
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    corrects = 0
    total = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        corrects += (predicted == labels).sum().item()
    
    train_loss = train_loss / len(train_loader.dataset)
    train_acc = corrects / total
    
    # Évaluation du modèle sur l'ensemble de validation
    model.eval()
    val_corrects = 0
    val_loss = 0.0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_corrects.double() / len(val_loader.dataset)
    
    # Ajout des données pour les courbes
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    
    print(f'Epoch [{epoch}/{num_epochs - 1}] '
          f'Train Loss: {train_loss:.4f} '
          f'Validation Loss: {val_loss:.4f} '
          f'Train Acc: {train_acc:.4f} '
          f'Validation Acc: {val_acc:.4f}')
    
    # Sauvegarde du modèle si la précision de validation est la meilleure jusqu'à présent
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model_VGG16.pth')

# Affichage des courbes de perte et d'exactitude
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Train')
plt.plot(range(num_epochs), val_losses, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(num_epochs), train_accuracies, label='Train Accuracy', color='blue')
plt.plot(range(num_epochs), val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
