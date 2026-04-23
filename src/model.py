from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_model(learning_rate=1e-4, num_classes=3):
    """
    Builds the deep learning architecture based on MobileNetV2.
    It takes an input of (224, 224, 3) which matches the pre-trained structure of ImageNet.
    We append our custom classification head (Dropout + Dense) for 3 classes:
    Proper, Improper, No Mask.
    """
    # Load the base model, discarding the fully connected top layers
    base_model = MobileNetV2(
        weights="imagenet", 
        include_top=False,
        input_tensor=Input(shape=(224, 224, 3))
    )

    # Freeze base model layers so we don't destroy pre-trained features during early epochs
    base_model.trainable = False

    # Construct the head of the model
    head_model = base_model.output
    head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
    head_model = Flatten(name="flatten")(head_model)
    
    # 128 Dense layer, followed by robust dropout to drastically minimize overfitting
    head_model = Dense(128, activation="relu")(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(num_classes, activation="softmax")(head_model)

    # Place the head FC model on top of the base model
    model = Model(inputs=base_model.input, outputs=head_model)

    # Compile our model
    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model

if __name__ == "__main__":
    # Smoke test
    model = build_model()
    model.summary()
