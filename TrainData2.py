import os
import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump

# Carica i dati grezzi
data = []
for i in range(22):
    data.append(pd.read_csv('../Dataset2/Network_dataset_' + str(1 + i) + '.csv', low_memory=False, nrows=100))

# Ottenere tutti i file CSV dalla cartella specificata
folder_path = '../RandomForest/Csv'  # Sostituisci con il percorso della tua cartella
csv_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

# Caricare i dati processati
processed_data = [pd.read_csv(csv_file, nrows=2300) for csv_file in csv_files]

# Estrarre i nomi dei file senza estensione per aggiungerli a columns_to_drop
file_names = [os.path.splitext(os.path.basename(csv_file))[0][5:] for csv_file in csv_files]

# Concatenare i DataFrame grezzi in uno unico
df = pd.concat(data, ignore_index=True)

# Rimuovere le colonne specificate
columns_to_drop = file_names
df = df.drop(columns=columns_to_drop, errors='ignore')
print(columns_to_drop)

# Aggiungere i dati processati come nuove colonne
for processed_df in processed_data:
    df = pd.concat([df, processed_df], axis=1)

df = df.drop(columns=columns_to_drop, errors='ignore')

df = df.drop(columns='ts', errors='ignore')
df = df.drop(columns='uid', errors='ignore')
df = df.drop(columns='dow', errors='ignore')

df.replace('-', 0, inplace=True)
df.replace('F', 0, inplace=True)
df.replace('T', 1, inplace=True)

X = df.drop('label', axis=1, errors='ignore')
y = df['label']
print(y)
y.fillna(y.mode()[0], inplace=True)
print(y)

non_numeric_columns = X.select_dtypes(include=['object']).columns
print(f"Colonne non numeriche: {non_numeric_columns}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.fillna(X_train.mean(), inplace=True)  # Sostituisce i NaN con la media
X_test.fillna(X_test.mean(), inplace=True)

model = TabNetClassifier()

# Addestramento
model.fit(
    X_train.values, y_train.values,  # Converti in array numpy
    eval_set=[(X_test.values, y_test.values)],  # Converti in array numpy
    eval_metric=['accuracy'],
    max_epochs=50,
    patience=10,
    batch_size=256,
    virtual_batch_size=128
)

# Mostra le prime righe per verificare il risultato
print(df.head())

y_pred = model.predict(X_test.values)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred))

# Salva il modello addestrato
dump(model, 'tabnetModel.joblib')