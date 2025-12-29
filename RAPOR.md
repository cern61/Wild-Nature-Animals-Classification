# 🦁 Vahşi Doğa Hayvanları Sınıflandırma Projesi
## CEREN NAZ DERVİŞOĞLU-23120205058

Bu proje, Derin Öğrenme teknikleri kullanılarak vahşi doğada bulunan 5 farklı hayvan türünü (**Aslan, Kuş, Timsah, Zebra, Zürafa**) sınıflandırmak amacıyla geliştirilmiştir. Proje, görüntü işleme ve yapay sinir ağları kullanılarak eğitilmiş bir model ve son kullanıcı için Gradio tabanlı bir web arayüzü içerir.

---

## 📋 İçindekiler
- [Projenin Amacı](#-projenin-amacı)
- [Literatür Taraması](#-literatür-taraması)
- [Veri Seti ve Ön İşleme](#-veri-seti-ve-ön-işleme)
- [Kullanılan Yöntem ve Algoritmalar](#-kullanılan-yöntem-ve-algoritmalar)
- [Model Mimarisi](#-model-mimarisi)
- [Eğitim ve Değerlendirme](#-eğitim-ve-değerlendirme)
- [Mevcut Çalışma vs Literatür](#-mevcut-çalışma-vs-literatür)

---

## 🎯 Projenin Amacı
Vahşi yaşam takibi, ekolojik dengenin korunması ve insan-vahşi yaşam çatışmalarının önlenmesi açısından kritiktir. Bu projenin temel amaçları şunlardır:
1.  Kamera tuzakları veya drone görüntülerinden elde edilen görselleri otomatik analiz etmek.
2.  Tehlikeli türler (Aslan, Timsah) ile av türlerini (Zebra, Zürafa) ayırt edebilen bir erken uyarı sistemi prototipi oluşturmak.
3.  Özel bir CNN mimarisi tasarlayarak özellik çıkarımı performansını ölçmek.

---

## 📚 Literatür Taraması
Bu alanda yapılan akademik çalışmalar incelendiğinde aşağıdaki yaklaşımlar öne çıkmaktadır:

1.  **Snapshot Serengeti (Norouzzadeh et al., 2018):** Milyonlarca kamera tuzağı görüntüsü ile 48 türü sınıflandırmışlardır. ResNet-50 mimarisi kullanılarak %96.6 doğruluk elde edilmiştir. Bu çalışma, alanın "altın standardı" olarak kabul edilir.
2.  **İHA Tabanlı Timsah Tespiti:** Avustralya'da yapılan çalışmalarda, su yüzeyindeki yansımalar nedeniyle zorlaşan timsah tespiti için YOLO nesne tespit modelleri kullanılmıştır.
3.  **Transfer Learning Yaklaşımları:** Literatürdeki çoğu çalışma, ImageNet ile eğitilmiş hazır modelleri (VGG16, MobileNet) kullanmaktadır.

---

## 💾 Veri Seti ve Ön İşleme
Projede özelleştirilmiş bir veri seti kullanılmıştır.
* **Sınıflar:** Aslan, Kuş, Timsah, Zebra, Zürafa.
* **Veri Kaynağı:** Açık kaynaklı görseller ve Google Images.
* **Veri Yapısı:** Eğitim(train) ve Doğrulama(val) olarak ikiye ayrılmıştır.

**Veri Çoğaltma (Data Augmentation):**
Modelin ezberlemesini (overfitting) önlemek ve vahşi doğa koşullarını simüle etmek için eğitim setine şu işlemler uygulanmıştır:
* Döndürme (Rotation)
* Yakınlaştırma (Zoom - Uzaktaki hayvanlar için)
* Kaydırma (Shift)
* Yatay Çevirme (Horizontal Flip)

---

## ⚙️ Kullanılan Yöntem ve Algoritmalar
Bu projede **Gözetimli Öğrenme (Supervised Learning)** yöntemi kullanılmıştır. Algoritma olarak, görüntü işlemede en başarılı yöntem olan **Evrişimli Sinir Ağları (Convolutional Neural Networks - CNN)** tercih edilmiştir.

Kullanılan Teknolojiler:
* **Dil:** Python 3.10
* **Kütüphaneler:** TensorFlow (Keras), Numpy, Matplotlib, Scikit-learn.
* **Arayüz:** Gradio.

---

## 🧠 Model Mimarisi
Hazır bir model kullanılmamış, problem için özel sıfırdan bir CNN mimarisi tasarlanmıştır.

| Katman Tipi | Özellikler | Açıklama |
| :--- | :--- | :--- |
| **Conv2D** | 32 Filtre, 3x3 | Temel kenar ve renk tespiti |
| **MaxPooling2D** | 2x2 | Boyut azaltma ve önemli özellikleri koruma |
| **Conv2D** | 64 Filtre, 3x3 | Doku ve şekil tespiti |
| **Conv2D** | 128 Filtre, 3x3 | Karmaşık obje parçalarının tespiti |
| **Flatten** | - | 2D matrisin vektöre çevrilmesi |
| **Dense** | 512 Nöron | Tam bağlantılı katman (Öğrenme) |
| **Dropout** | 0.5 | Rastgele nöron kapatma (Overfitting önleyici) |
| **Dense (Output)** | 5 Nöron (Softmax) | Sınıflandırma olasılıkları |

---

## 📊 Eğitim ve Değerlendirme
Model, **Categorical Crossentropy** kayıp fonksiyonu ve **Adam** optimizasyon algoritması ile eğitilmiştir.

### Başarı Metrikleri
Modelin performansı aşağıdaki metriklerle ölçülmüştür:
* **Doğruluk (Accuracy):** Genel başarı oranı.
* **Karmaşıklık Matrisi (Confusion Matrix):** Hangi hayvanın hangi hayvanla karıştırıldığının analizi.
* **F1-Skoru:** Dengesiz veri dağılımlarına karşı hassasiyet ölçümü.

![alt text](karmasiklik_matrisi.png)


              precision    recall  f1-score   support

       aslan       1.00      1.00      1.00         1
         kus       1.00      1.00      1.00         1
      timsah       1.00      1.00      1.00         1
       zebra       1.00      1.00      1.00         1
      zurafa       1.00      1.00      1.00         1

    accuracy                           1.00         5
   macro avg       1.00      1.00      1.00         5
weighted avg       1.00      1.00      1.00         5



---

## 🆚 Mevcut Çalışma vs Literatür

| Özellik | Literatürdeki Genel Çalışmalar | Bizim Projemiz |
| :--- | :--- | :--- |
| **Model Tipi** | ResNet, VGG16 (Ağır Modeller) | Özel Tasarım Hafif CNN |
| **Veri Boyutu** | Milyonlarca Görüntü | Odaklanmış, Küçük Veri Seti |
| **Donanım** | GPU Cluster / Sunucu | Standart CPU/GPU (Erişilebilir) |
| **Amaç** | Genel Biyoçeşitlilik Sayımı | Hızlı Prototipleme & Eğitim Amaçlı |
| **Kullanım** | Bilimsel Analiz | Son Kullanıcı Arayüzü (Gradio) |

**Sonuç:** Bizim projemiz, devasa kaynaklara ihtiyaç duymadan, belirli bir bölgedeki hedef türleri tanımak için optimize edilmiş, taşınabilir ve hızlı bir çözüm sunmaktadır.

---
