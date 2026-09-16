# Obezite Seviyesi Sınıflandırma ve Tahmin Portalı

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)

Bu proje, insanların fiziksel ölçümleri, beslenme alışkanlıkları ve yaşam tarzı faktörlerine dayanarak obezite seviyelerini tahmin eden **PyTorch Derin Öğrenme Modeli** ve bu model ile etkileşim kuran **Modern Web Uygulaması**'dır.

---

## Proje Hakkında

Obezite, dünya genelinde yaygınlaşan ciddi bir sağlık sorunudur. Bu çalışmada, Kaggle Obezite Riski veri seti (`train.csv`) kullanılarak 7 farklı obezite seviyesini tahmin eden bir Derin Yapay Sinir Ağı (Deep Neural Network) eğitilmiştir.

Eğitilen PyTorch modeli (`obesity_classifier.pth`), ön işleme pipeline (`ColumnTransformer`) ve modern web arayüzü entegre edilerek kullanıcı dostu bir web portalına dönüştürülmüştür.

---

##  Model Mimarisi & Ön İşleme

### 1. Ön İşleme Adımları (Preprocessing)
- **İkili Kategorik Değişkenler (`OrdinalEncoder`)**: `family_history_with_overweight`, `FAVC`, `SMOKE`, `SCC`
- **Sıralı Kategorik Değişkenler (`OrdinalEncoder`)**: `CAEC`, `CALC` (no < Sometimes < Frequently < Always)
- **Nominal Kategorik Değişkenler (`OneHotEncoder`)**: `Gender`, `MTRANS` (drop='first')
- **Sayısal Değişkenler (`StandardScaler`)**: `Age`, `Height`, `Weight`, `FCVC`, `NCP`, `CH2O`, `FAF`, `TUE`

### 2. PyTorch Derin Öğrenme Yapısı (`ObesityClassifier`)
```
Girdi (19 Öznitelik) 
   ↓
Linear(19 → 64) + ReLU + Dropout(0.2)
   ↓
Linear(64 → 32) + ReLU + Dropout(0.2)
   ↓
Linear(32 → 16) + ReLU + Dropout(0.2)
   ↓
Linear(16 → 7) → Logits / Softmax (7 Obezite Sınıfı)
```

### 3. Hedef Obezite Sınıfları
1. `Insufficient_Weight` (Yetersiz Kilo / Zayıf)
2. `Normal_Weight` (Normal Kilo)
3. `Overweight_Level_I` (1. Derece Fazla Kilo)
4. `Overweight_Level_II` (2. Derece Fazla Kilo)
5. `Obesity_Type_I` (1. Derece Obezite)
6. `Obesity_Type_II` (2. Derece Obezite)
7. `Obesity_Type_III` (3. Derece Obezite / Morbid)

---

## Web Uygulaması Özellikleri

-  **Modern Koyu Tema & Cam Efekti (Dark Glassmorphic UI)**: Şık görsel tasarım ve akıcı animasyonlar.
-  **Etkileşimli Form Sürgüleri**: Yaş, boy, kilo, su miktarı ve egzersiz değerleri için anlık birim göstergeleri.
-  **Animasyonlu Olasılık Dağılımı**: Modelin 7 obezite sınıfı için hesapladığı % olasılık değerlerinin grafik gösterimi.
-  **Vücut Kitle İndeksi (VKİ / BMI)**: Otomatik hesaplama ve durum göstergesi.
-  **Kişiselleştirilmiş Sağlık Tavsiyeleri**: Kullanıcının su tüketimi, hareket seviyesi ve beslenme tercihlerine özel yaşam tarzı önerileri.

---

## Proje Dosya Yapısı

```
.
├── server.py               # Python HTTP Web Sunucusu ve API servisi
├── test_model.py           # Model doğrulama ve test betiği
├── obesity_classifier.pth  # Eğitilmiş PyTorch model ağırlıkları
├── train.csv               # Eğitim ve ön işleme veri seti
├── Untitled14 (10).ipynb   # Model eğitimi ve analiz Jupyter Notebook'u
├── requirements.txt        # Gerekli Python kütüphaneleri
├── .gitignore              # Git tarafından yoksayılacak dosyalar
├── README.md               # Proje dokümantasyonu
└── static/                 # Web ön yüzü statik dosyaları
    ├── index.html          # HTML5 sayfa yapısı
    ├── style.css           # CSS3 stil dosyası
    └── script.js           # JavaScript etkileşim mantığı
```

---

## Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/KULLANICI_ADI/Obezite-Siniflandirma.git
cd Obezite-Siniflandirma
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Web Sunucusunu Başlatın
```bash
python server.py
```

### 4. Tarayıcıda Açın
Sunucu başladığında tarayıcınızdan şu adrese gidin:
```
http://localhost:5000
```

---
