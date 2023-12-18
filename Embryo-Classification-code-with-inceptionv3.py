import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3  # Utilisez InceptionV3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd

# Spécifiez le répertoire de données
data_dir = "./data-set-embr/train"  # Le dossier "train" contenant les sous-dossiers de classe

# Dimensions d'entrée de vos images RVB (à adapter en fonction de vos besoins)
input_shape = (224, 224, 3)

# Créez un dictionnaire pour mapper les noms de classe aux étiquettes de classe
class_name_to_label = {
    "Rien": 0,
    "t2": 1,
    "tB": 2,
    "tPB2": 3,
    "tPNa": 4
}

# Créez des listes pour stocker les images et les étiquettes
X = []  # Pour les images
y = []  # Pour les étiquettes (labels)

# Parcourez les dossiers de classe
for class_name in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(class_dir):
        class_label = class_name_to_label.get(class_name, -1)  # Utilisez -1 comme étiquette par défaut si le nom du dossier n'est pas trouvé dans le dictionnaire

        if class_label != -1:  # Vérifiez si l'étiquette est valide
            # Parcourez les sous-dossiers à l'intérieur du dossier de classe
            for sub_folder in os.listdir(class_dir):
                sub_folder_path = os.path.join(class_dir, sub_folder)
                if os.path.isdir(sub_folder_path):
                    # Parcourez les fichiers d'images à l'intérieur du sous-dossier
                    for image_name in os.listdir(sub_folder_path):
                        image_path = os.path.join(sub_folder_path, image_name)
                        image = cv2.imread(image_path)

                        if image is not None:
                            # Redimensionnez l'image pour correspondre aux dimensions d'entrée de InceptionV3
                            image = cv2.resize(image, (input_shape[0], input_shape[1]))
                            X.append(image)
                            y.append(class_label)

# Convertissez les listes en tableaux NumPy
X = np.array(X)
y = np.array(y)

# Divisez les données en ensembles d'entraînement et de validation (à adapter en fonction de vos besoins)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Créez un générateur d'images pour l'ensemble de données d'entraînement avec augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Créez le modèle InceptionV3
base_model = InceptionV3(  # Utilisez InceptionV3
    include_top=False,
    weights='imagenet',
    input_shape=input_shape,
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(len(class_name_to_label), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compilez le modèle
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
model.summary()

# Configurez des callbacks pour l'entraînement
checkpoint_callback = ModelCheckpoint('Embryo_classification_model_InceptionV3.h5', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Entraînez le modèle
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=15,
    validation_data=valid_datagen.flow(X_valid, y_valid, batch_size=32),  # Assurez-vous que votre validation_data est bien spécifié
    validation_steps=len(X_valid) // 32,
    callbacks=[early_stopping, checkpoint_callback]
)

# Créez un dataframe pandas à partir des résultats d'entraînement
history_df = pd.DataFrame(history.history)

# Enregistrez le dataframe dans un fichier Excel
history_df.to_excel("Embryo_training_results_InceptionV3.xlsx", index=False)

# Affichez des courbes d'apprentissage
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['acc'], label='Train Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.show()
