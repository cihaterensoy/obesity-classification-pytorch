import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder

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
model.load_state_dict(torch.load('obesity_classifier.pth', map_location='cpu'))
model.eval()

sample = X.iloc[[0]]
sample_proc = preprocessor.transform(sample)
logits = model(torch.tensor(sample_proc, dtype=torch.float32))
probs = torch.softmax(logits, dim=1).detach().numpy()[0]

print("Classes:", list(le.classes_))
print("Sample input:", sample.to_dict(orient='records')[0])
print("Predicted class:", le.classes_[probs.argmax()], "Actual class:", y.iloc[0])
print("Probabilities:")
for cls, prob in zip(le.classes_, probs):
    print(f"  {cls}: {prob*100:.2f}%")
