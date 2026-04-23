import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from model import build_model
from data_preprocessing import DataPreprocessor

def train():
    dataset_path = "../dataset"
    if not os.path.exists(dataset_path) or not os.listdir(dataset_path):
        print("WARNING: Dataset folder is empty! Please drop the organized Kaggle dataset into the 'dataset' directory.")
        print("Expected structure: dataset/Mask_Proper/ , dataset/Mask_Improper/ , dataset/No_Mask/")
        return

    # 1. Initialize data loaders
    preprocessor = DataPreprocessor(dataset_path=dataset_path, img_size=(224, 224), batch_size=32)
    train_generator, val_generator = preprocessor.get_data_generators()

    # 2. Build model architecture
    model = build_model(learning_rate=1e-4, num_classes=3)
    
    # 3. Setup Callbacks
    os.makedirs("../models", exist_ok=True)
    checkpoint = ModelCheckpoint(
        "../models/mask_detector.h5", 
        monitor="val_loss", 
        save_best_only=True, 
        verbose=1
    )
    # Stop early if the loss starts plateauing and divergence happens (overfitting)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

    epochs = 20

    print("[INFO] Starting Neural Network Training...")
    # 4. Train
    H = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=epochs,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 5. Plot training history
    print("[INFO] Saving training graphs...")
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(H.history["loss"], label="train_loss")
    plt.plot(H.history["val_loss"], label="val_loss")
    plt.plot(H.history["accuracy"], label="train_acc")
    plt.plot(H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("../models/training_history.png")
    
if __name__ == "__main__":
    train()
