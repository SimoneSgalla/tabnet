import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pytorch_tabnet.tab_model import TabNetClassifier
import numpy as np
import os


def load_and_concatenate_csv(directory_path, strategy="mean", chunksize=100000, fill_value=0):
    """
    Carica e concatena tutti i file CSV presenti in una directory.

    Parametri:
        directory_path (str): Percorso della directory contenente i file CSV.

    Ritorna:
        pd.DataFrame: Un DataFrame contenente i dati concatenati.
    """
    all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]

    # Lista per memorizzare i DataFrame
    dataframes = []
    labels = []

    for i, file in enumerate(all_files):
        print(f"Caricando {file}...")
        #data = dd.read_csv(file)
        chunks = pd.read_csv(file, chunksize=chunksize)
        for chunk in chunks:
            if i == 0:
                labels.append(chunk['Label'])
                chunk = chunk.drop(columns=['Timestamp', 'Label'])
            else:
                labels.append(chunk['Label'])
                chunk = chunk.drop(columns=['Timestamp', 'Label'])
                chunk = chunk.iloc[1:]
            chunk = chunk.replace([np.inf, -np.inf], np.nan)

            # Gestione dei NaN
            if strategy == "mean":
                chunk = chunk.fillna(chunk.mean())
            elif strategy == "median":
                chunk = chunk.fillna(chunk.median())
            elif strategy == "constant":
                chunk = chunk.fillna(fill_value)
            else:
                raise ValueError("Strategia non valida. Usa 'mean', 'median' o 'constant'.")

            dataframes.append(chunk)

    concatenated_data = pd.concat(dataframes, ignore_index=True)
    concatenated_labels = pd.concat(labels, ignore_index=True)

    return concatenated_data, concatenated_labels


# Caricamento dataset

data, y = load_and_concatenate_csv('../Dataset1', strategy="constant")
print('ciao')

X = data

print(X.columns)

# Codifica del target se necessario (esempio: classificazione con etichette non numeriche)
le = LabelEncoder()
y = le.fit_transform(y)

# Divisione in train e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Conversione in array numpy (richiesto da TabNet)
X_train = X_train.values
X_test = X_test.values
y_train = y_train
y_test = y_test

# Inizializzazione del modello TabNet (classificazione)
model = TabNetClassifier()

# Addestramento
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['accuracy'],
    max_epochs=50,
    patience=10,
    batch_size=256,
    virtual_batch_size=128
)

# Valutazione
accuracy = model.predict(X_test)
print(f"Accuracy: {np.mean(accuracy == y_test):.2f}")

# Salvare il modello
model.save_model('tabnet_model')
