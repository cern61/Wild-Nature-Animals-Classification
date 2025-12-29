import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = os.path.dirname(os.path.abspath(__file__))
val_dir = os.path.join(base_dir, 'dataset', 'val')
model_path = 'vahsi_yasam_model.keras'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

if not os.path.exists(model_path):
    print("HATA: Model dosyası bulunamadı! Önce 'egitim.py' çalıştırılmalı.")
    exit()

print("Model yükleniyor.")
model = tf.keras.models.load_model(model_path)

print("Test verileri hazırlanıyor.")
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False 
)

class_names = list(val_generator.class_indices.keys())

print("Model test verileri üzerinde tahmin yapıyor...")

Y_pred = model.predict(val_generator)

y_pred = np.argmax(Y_pred, axis=1)

y_true = val_generator.classes

#KARMAŞIKLIK MATRİSİ
print("\nGrafik çiziliyor.")
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Karmaşıklık Matrisi (Confusion Matrix)')
plt.ylabel('Gerçek Tür')
plt.xlabel('Modelin Tahmini')
plt.savefig('karmasiklik_matrisi.png')
print("1. 'karmasiklik_matrisi.png' kaydedildi.")



print("\n--- DETAYLI PERFORMANS RAPORU ---")
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)


with open("basari_raporu.txt", "w") as f:
    f.write(report)
print("2. 'basari_raporu.txt' dosyasına istatistikler yazıldı.")