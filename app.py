import gradio as gr
import tensorflow as tf
import numpy as np
import os


model_path = 'vahsi_yasam_model.keras'

if not os.path.exists(model_path):
    print("HATA: Model dosyası bulunamadı! Önce 'egitim.py' dosyasını çalıştırıp modeli eğitmelisin.")
    exit()

print("Model yükleniyor, lütfen bekleyin.")
model = tf.keras.models.load_model(model_path)

class_names = ['aslan', 'kus', 'timsah', 'zebra', 'zurafa']


def tahmin_et(img):
    if img is None:
        return None
 
    img = tf.image.resize(img, (224, 224))
    
    img = img / 255.0
    
    img = np.expand_dims(img, axis=0)
    
    predictions = model.predict(img).flatten()
    
    
    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🦁 Vahşi Yaşam Sınıflandırma Projesi
    Bu yapay zeka modeli; **Aslan, Kuş, Timsah, Zebra ve Zürafa** fotoğraflarını ayırt edebilir.
    """)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Fotoğraf Yükle veya Çek")
            predict_btn = gr.Button("Analiz Et 🚀", variant="primary")
        
        with gr.Column():
            output_label = gr.Label(num_top_classes=3, label="Derin Ogrenme Tahmini:")
    
    
    predict_btn.click(fn=tahmin_et, inputs=input_image, outputs=output_label)


print("Arayüz başlatılıyor.")
demo.launch()