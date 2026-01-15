"""
Script d'entraînement du modèle CNN pour la reconnaissance du langage des signes ASL
Auteur: Expert IA
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Configuration (Optimisée pour un entraînement plus rapide)
IMG_SIZE = 64  # Réduit de 128 à 64 pour accélérer (2x plus rapide)
BATCH_SIZE = 64  # Augmenté de 32 à 64 pour traiter plus d'images à la fois
EPOCHS = 30  # Réduit de 50 à 30 (souvent suffisant avec EarlyStopping)
DATASET_PATH = 'dataset/train'
MODEL_SAVE_PATH = 'model/asl_model.h5'

def create_cnn_model(num_classes):
    """
    Crée un modèle CNN optimisé pour la reconnaissance des signes ASL
    
    Args:
        num_classes: Nombre de classes à prédire
    
    Returns:
        model: Modèle Keras compilé
    """
    model = keras.Sequential([
        # Bloc 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloc 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloc 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloc 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Couches denses
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilation du modèle
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_data():
    """
    Prépare les générateurs de données avec augmentation
    
    Returns:
        train_generator: Générateur pour les données d'entraînement
        class_names: Liste des noms de classes
    """
    # Augmentation de données pour améliorer la généralisation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        validation_split=0.2  # 20% pour la validation
    )
    
    # Générateur d'entraînement
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Générateur de validation
    validation_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator, list(train_generator.class_indices.keys())

def plot_training_history(history):
    """
    Affiche les courbes d'apprentissage
    
    Args:
        history: Historique d'entraînement Keras
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Courbe de précision
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Précision du modèle')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Courbe de perte
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Perte du modèle')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('model/training_history.png', dpi=300, bbox_inches='tight')
    print("📊 Graphiques sauvegardés dans model/training_history.png")
    plt.show()

def main():
    """
    Fonction principale d'entraînement
    """
    print("🚀 Démarrage de l'entraînement du modèle ASL...")
    
    # Créer le dossier model s'il n'existe pas
    os.makedirs('model', exist_ok=True)
    
    # Préparer les données
    print("\n📂 Chargement et préparation des données...")
    train_gen, val_gen, class_names = prepare_data()
    
    num_classes = len(class_names)
    print(f"✅ Nombre de classes détectées: {num_classes}")
    print(f"📋 Classes: {class_names}")
    print(f"📊 Images d'entraînement: {train_gen.samples}")
    print(f"📊 Images de validation: {val_gen.samples}")
    
    # Créer le modèle
    print("\n🏗️ Construction du modèle CNN...")
    model = create_cnn_model(num_classes)
    model.summary()
    
    # Callbacks pour l'entraînement
    callbacks = [
        # Sauvegarde du meilleur modèle
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Arrêt anticipé si pas d'amélioration
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Réduction du taux d'apprentissage
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Entraînement
    print("\n🎯 Début de l'entraînement...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarde des noms de classes
    class_names_path = 'model/class_names.txt'
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"\n✅ Noms des classes sauvegardés dans {class_names_path}")
    
    # Affichage des résultats
    print("\n📈 Résultats finaux:")
    print(f"   Précision d'entraînement: {history.history['accuracy'][-1]:.4f}")
    print(f"   Précision de validation: {history.history['val_accuracy'][-1]:.4f}")
    
    # Afficher les courbes d'apprentissage
    plot_training_history(history)
    
    print(f"\n✅ Modèle sauvegardé avec succès dans {MODEL_SAVE_PATH}")
    print("🎉 Entraînement terminé!")

if __name__ == "__main__":
    main()