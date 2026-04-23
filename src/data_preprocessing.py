import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataPreprocessor:
    def __init__(self, dataset_path, img_size=(224, 224), batch_size=32):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size

    def get_data_generators(self):
        """
        Builds the training and validation data generators with data augmentation.
        Augmentation is crucial to prevent overfitting and robustly train the model 
        across varying angles, lighting, and zoom levels, giving it a real-world edge.
        """
        # Rescale pixel values to [-1, 1] as expected by MobileNetV2
        # Apply augmentation for train datagen
        train_datagen = ImageDataGenerator(
            preprocessing_function=lambda x: (x / 127.5) - 1.0,
            rotation_range=20,
            zoom_range=0.15,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.15,
            horizontal_flip=True,
            fill_mode="nearest",
            validation_split=0.2 # 80/20 train/val split
        )

        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        return train_generator, val_generator

if __name__ == "__main__":
    # Test block
    print("Data preprocessor module initialized.")
