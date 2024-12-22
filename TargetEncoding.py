# Carica il dataset
import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.preprocessing import TargetEncoder

category = 'Timestamp'

directory_path = '../Dataset1'

data = []

all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]

for file in all_files:
    data.append(pd.read_csv(file, usecols=[category]))

df2 = pd.read_csv('Csv/TransLabel.csv', low_memory=False)

df = pd.concat(data)
print(df)

X = df
y = df2

y = y.values.ravel()

enc_auto = TargetEncoder(smooth="auto")
X_trans = enc_auto.fit_transform(X, y)

X_trans_flat = X_trans.ravel()

# Crea un DataFrame che include sia gli IP originali sia la colonna codificata
encoded_df = pd.DataFrame({
    category: X[category],
    'encoded_value': X_trans_flat
})

# Salva il DataFrame completo nel file CSV
encoded_df.to_csv('Csv/Trans'+category+'.csv', index=False)

