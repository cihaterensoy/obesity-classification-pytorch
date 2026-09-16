import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder

# --- 1. MODEL & PREPROCESSOR SETUP ---
print("Model ve veri önişlemcisi yükleniyor...")
train = pd.read_csv('train.csv')

binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
ordinal_cols = ['CAEC', 'CALC']
nominal_cols = ['Gender', 'MTRANS']
numeric_cols = [col for col in train.select_dtypes(include="number").columns if col not in ['id', 'NObeyesdad']]

preprocessor = ColumnTransformer(
    transformers=[
        ("binary", OrdinalEncoder(), binary_cols),
        ("ordinal", OrdinalEncoder(
            categories=[
                ['no', 'Sometimes', 'Frequently', 'Always'],
                ['no', 'Sometimes', 'Frequently', 'Always']
            ]
        ), ordinal_cols),
        ("nominal", OneHotEncoder(drop="first", handle_unknown="infrequent_if_exist"), nominal_cols),
        ('numeric', StandardScaler(), numeric_cols)
    ]
)

X = train.drop(columns=['id', 'NObeyesdad'])
y = train['NObeyesdad']

le = LabelEncoder()
le.fit(y)
preprocessor.fit(X)

class ObesityClassifier(nn.Module):
    def __init__(self, in_features=19, out_features=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=32, out_features=16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=16, out_features=out_features)
        )
    def forward(self, x):
        return self.network(x)

model = ObesityClassifier()
model_path = 'obesity_classifier.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    print("Model ağırlıkları başarıyla yüklendi.")
else:
    print(f"Hata: {model_path} bulunamadı!")
model.eval()

CLASS_TRANSLATIONS = {
    "Insufficient_Weight": "Yetersiz Kilo (Zayıf)",
    "Normal_Weight": "Normal Kilo",
    "Overweight_Level_I": "Fazla Kilo Seviye I",
    "Overweight_Level_II": "Fazla Kilo Seviye II",
    "Obesity_Type_I": "Obezite Tip I",
    "Obesity_Type_II": "Obezite Tip II",
    "Obesity_Type_III": "Obezite Tip III (Morbid Obezite)"
}

CLASS_DESCRIPTIONS = {
    "Insufficient_Weight": "Vücut kitle indeksiniz standart değerlerin altında. Dengeli ve kalori açısından zengin beslenme ile ideal kilonuza ulaşabilirsiniz.",
    "Normal_Weight": "Tebrikler! Vücut kitle indeksiniz ve yaşam tarzınız sağlıklı sınırlar içerisinde.",
    "Overweight_Level_I": "Kilonuz ideal aralığın biraz üzerinde. Düzenli egzersiz ve hafif diyet düzenlemeleri faydalı olabilir.",
    "Overweight_Level_II": "Kilonuz ideal seviyenin belirgin şekilde üzerinde. Aktif yaşam tarzı ve porsiyon kontrolü önem taşımaktadır.",
    "Obesity_Type_I": "Birinci derece obezite kategorisindesiniz. Beslenme uzmanı desteği ve haftalık düzenli egzersiz programı önerilir.",
    "Obesity_Type_II": "İkinci derece obezite kategorisindesiniz. Sağlıklı yaşam alışkanlıkları ve tıbbi/beslenme danışmanlığı almanız faydalı olacaktır.",
    "Obesity_Type_III": "Üçüncü derece (morbid) obezite kategorisindesiniz. En kısa sürede uzman bir hekim ve diyetisyen eşliğinde multidisipliner destek almanız önerilir."
}

def generate_insights(data, predicted_class, bmi):
    insights = []
    
    # BMI insight
    if bmi < 18.5:
        insights.append("💡 VKİ değeriniz 18.5 altında. Kas kütlenizi artırmaya yönelik kuvvet egzersizleri ve besleyici gıdalar tüketebilirsiniz.")
    elif bmi < 25:
        insights.append("✅ VKİ değeriniz ideal aralıkta (18.5 - 24.9). Mevcut beslenme ve aktivite düzeninizi koruyun.")
    elif bmi < 30:
        insights.append("⚠️ VKİ değeriniz 25 - 29.9 aralığında. Kilo artışını önlemek için günlük hareket miktarınızı artırabilirsiniz.")
    else:
        insights.append("❗ VKİ değeriniz 30 ve üzerinde. Kardiyovasküler sağlık için kilo yönetimi öncelikli hale getirilmelidir.")

    # Water insight
    ch2o = float(data.get('CH2O', 2))
    if ch2o < 2:
        insights.append("💧 Günlük su tüketiminiz düşük (< 2 Litre). Metabolizmayı hızlandırmak ve sağlığınızı korumak için günde en az 2.5-3 litre su içmeye özen gösterin.")
    else:
        insights.append("💧 Su tüketim seviyeniz harika! Yeterli hidrasyon metabolizma sağlığını destekler.")

    # Physical activity
    faf = float(data.get('FAF', 1))
    if faf == 0:
        insights.append("🏃 Fiziksel aktivite seviyeniz çok düşük. Günde en az 30 dakikalık tempolu yürüyüşlerle başlamanız önerilir.")
    elif faf >= 2:
        insights.append("💪 Düzenli fiziksel aktivite yapıyorsunuz, tebrikler! Bu alışkanlığı sürdürmek kas kütlenizi korur.")

    # High calorie food / snacks
    favc = data.get('FAVC', 'no')
    caec = data.get('CAEC', 'no')
    if favc == 'yes' or caec in ['Frequently', 'Always']:
        insights.append("🥗 Yüksek kalorili besin tüketimi veya sık atıştırma alışkanlığınız var. İşlenmiş gıdalar yerine lifli ve taze gıdaları tercih edebilirsiniz.")

    # Tech device usage
    tue = float(data.get('TUE', 0))
    if tue >= 2:
        insights.append("📱 Günlük ekran süreniz yüksek. Hareketsiz kalma sürenizi azaltmak için saat başı kısa molalar verip esneme hareketleri yapabilirsiniz.")

    return insights

# --- 2. HTTP REQUEST HANDLER ---
class PredictHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory='static', **kwargs)

    def do_GET(self):
        if self.path == '/api/info':
            info_data = {
                "categories": {
                    "Gender": ["Female", "Male"],
                    "family_history_with_overweight": ["no", "yes"],
                    "FAVC": ["no", "yes"],
                    "CAEC": ["no", "Sometimes", "Frequently", "Always"],
                    "SMOKE": ["no", "yes"],
                    "SCC": ["no", "yes"],
                    "CALC": ["no", "Sometimes", "Frequently"],
                    "MTRANS": ["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
                },
                "classes": list(le.classes_),
                "translations": CLASS_TRANSLATIONS
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.end_headers()
            self.wfile.write(json.dumps(info_data, ensure_ascii=False).encode('utf-8'))
            return
        
        # Default static file handling
        return super().do_GET()

    def do_POST(self):
        if self.path == '/api/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            try:
                data = json.loads(body.decode('utf-8'))
                
                # Height conversion if cm was passed
                height = float(data.get('Height', 1.70))
                if height > 3.0: # passed in cm
                    height = height / 100.0
                
                weight = float(data.get('Weight', 70.0))
                age = float(data.get('Age', 25.0))
                fcvc = float(data.get('FCVC', 2.0))
                ncp = float(data.get('NCP', 3.0))
                ch2o = float(data.get('CH2O', 2.0))
                faf = float(data.get('FAF', 1.0))
                tue = float(data.get('TUE', 1.0))

                input_dict = {
                    'Gender': str(data.get('Gender', 'Male')),
                    'Age': age,
                    'Height': height,
                    'Weight': weight,
                    'family_history_with_overweight': str(data.get('family_history_with_overweight', 'yes')),
                    'FAVC': str(data.get('FAVC', 'yes')),
                    'FCVC': fcvc,
                    'NCP': ncp,
                    'CAEC': str(data.get('CAEC', 'Sometimes')),
                    'SMOKE': str(data.get('SMOKE', 'no')),
                    'CH2O': ch2o,
                    'SCC': str(data.get('SCC', 'no')),
                    'FAF': faf,
                    'TUE': tue,
                    'CALC': str(data.get('CALC', 'Sometimes')),
                    'MTRANS': str(data.get('MTRANS', 'Public_Transportation'))
                }

                sample_df = pd.DataFrame([input_dict])
                sample_proc = preprocessor.transform(sample_df)
                
                with torch.no_grad():
                    logits = model(torch.tensor(sample_proc, dtype=torch.float32))
                    probs = torch.softmax(logits, dim=1).numpy()[0]
                
                top_idx = probs.argmax()
                predicted_class = str(le.classes_[top_idx])
                
                # BMI Calculation: weight / (height_m ^ 2)
                bmi = round(weight / (height ** 2), 2)
                
                probs_dict = []
                for cls, prob in zip(le.classes_, probs):
                    probs_dict.append({
                        "class_code": cls,
                        "class_tr": CLASS_TRANSLATIONS.get(cls, cls),
                        "probability": round(float(prob) * 100, 2)
                    })
                
                # Sort probabilities descending
                probs_dict.sort(key=lambda x: x["probability"], reverse=True)
                
                response_data = {
                    "success": True,
                    "predicted_class": predicted_class,
                    "predicted_class_tr": CLASS_TRANSLATIONS.get(predicted_class, predicted_class),
                    "description": CLASS_DESCRIPTIONS.get(predicted_class, ""),
                    "bmi": bmi,
                    "probabilities": probs_dict,
                    "insights": generate_insights(input_dict, predicted_class, bmi)
                }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                self.wfile.write(json.dumps(response_data, ensure_ascii=False).encode('utf-8'))
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.send_response(400)
                self.send_header('Content-Type', 'application/json; charset=utf-8')
                self.end_headers()
                err_resp = {"success": False, "error": str(e)}
                self.wfile.write(json.dumps(err_resp, ensure_ascii=False).encode('utf-8'))
            return
            
        self.send_response(404)
        self.end_headers()

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def run(server_class=HTTPServer, handler_class=PredictHandler, port=5000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f"Obezite Sınıflandırma Web Sunucusu Başlatıldı: http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu kapatılıyor...")
        httpd.server_close()

if __name__ == '__main__':
    run()

