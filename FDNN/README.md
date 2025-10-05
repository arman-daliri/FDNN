# FDNN

A lightweight implementation of a **Fuzzy Deep Neural Network (FDNN)** in TensorFlow.

## Installation
```bash
pip install FDNN
```

## Usage
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import FDNN as fd

# Load dataset
df = pd.read_csv("MAINDiagnostics.csv")
df["Gender"] = df["Gender"].map({"MALE": 1, "FEMALE": 0})
df = df.drop(columns=["IDFILENAME", "FileName", "Beat"], errors='ignore')

# Create binary target
normal_group = ["SR", "SB", "ST", "SI", "SAAWR"]
arrhythmia_group = ["AFIB", "AF", "SVT", "AT", "AVNRT", "AVRT"]
df["Rhythm_Binary"] = df["Rhythm"].apply(lambda x: 0 if x in normal_group else (1 if x in arrhythmia_group else None))
df = df.dropna(subset=["Rhythm_Binary"])
y = df["Rhythm_Binary"].astype(int)
X = df.drop(columns=["Rhythm", "Rhythm_Binary"], errors='ignore')

# Save feature names
feature_names = X.columns.tolist()

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
input_dim = X_train.shape[1]

# Train the model
model = fd.build_fdnn(input_dim)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32, verbose=1)
```
You can find "MAINDiagnostics.csv" [here](https://github.com/arman-daliri/FDNN/blob/main/MAINDiagnostics.csv).

## License
MIT

## Authors
- Arman Daliri
- Nora Mahdavi
