import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

# Parameters
img_size = 64
batch_size = 32
epochs = 50
dataset_path = 'dataset/'

# Data generators with improved augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,        # Reduced
    width_shift_range=0.1,    # Reduced
    height_shift_range=0.1,   # Reduced
    shear_range=0.1,          # Reduced
    zoom_range=0.1,           # Reduced
    horizontal_flip=False     # ❗ CRITICAL CHANGE: Disabled horizontal flip
)

train_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Save the class indices to a JSON file for use in the main script
labels = train_data.class_indices
# Invert the dictionary to map index to label name
labels = {v: k for k, v in labels.items()}
with open('labels.json', 'w') as f:
    json.dump(labels, f)
print("✅ Class labels saved to labels.json")

# CNN model (deeper, with BatchNorm + Dropout)
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(512, activation='relu'),
    Dropout(0.5),

    Dense(train_data.num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train model
history = model.fit(train_data, validation_data=val_data, epochs=epochs)

# Save trained model
model.save('model01_cnn.h5')
print("✅ Model saved as model01_cnn.h5")