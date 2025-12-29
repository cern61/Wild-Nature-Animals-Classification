import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

base_dir = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(base_dir, 'dataset', 'train')
val_dir = os.path.join(base_dir, 'dataset', 'val')


IMG_SIZE = (224, 224)
BATCH_SIZE = 8  
EPOCHS = 10     

print(f"Eğitim verisi aranıyor: {train_dir}")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("Veriler yükleniyor.")
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


class_names = list(train_generator.class_indices.keys())
print(f"Tespit edilecek sınıflar: {class_names}")


model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Model eğitimi başlıyor.")
history = model.fit(train_generator, epochs=EPOCHS, validation_data=val_generator)

model.save('vahsi_yasam_model.keras')
print("TEBRİKLER! Model eğitildi ve 'vahsi_yasam_model.keras' olarak kaydedildi.")