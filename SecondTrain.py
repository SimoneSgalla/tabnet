from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Carica il modello salvato
model = TabNetClassifier()
model.load_model("tabnet_model8.zip")

# Carica e prepara il nuovo dataset
new_data = pd.read_csv("../Dataset1/Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv", low_memory=False)
new_data = new_data.replace([np.inf, -np.inf], np.nan)
non_numeric_columns = new_data.select_dtypes(include=['object', 'category']).columns

for col in non_numeric_columns:
    try:
        new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
    except Exception as e:
        print(f"Errore nella conversione di {col}: {e}")

new_data = new_data.fillna(0)  # Sostituisci con la strategia di preprocessamento usata in precedenza

# Separa le feature (X) e le label (y)
new_data = new_data.drop(columns=["Timestamp"])
y_new = new_data["Label"]
X_new = new_data.drop(columns=["Label"])
X_new = X_new.clip(lower=-1e10, upper=1e10)
X = X_new.fillna(X_new.mean())


# Converte le etichette in numeri se necessario (ad esempio, categorizzazione)
encoder = LabelEncoder()
y_new = encoder.fit_transform(y_new)

# Ri-Addestra il modello
'''model.fit(
    X_train=X_new.values,  # Assicurati di convertire in NumPy array
    y_train=y_new,
    max_epochs=50,         # Numero di epoche da configurare
    patience=10,           # Early stopping
    batch_size=1024,       # Dimensione del batch
    virtual_batch_size=128,  # Virtual batch size
    from_unsupervised=model,  # Indica che è supervisato
)'''

X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

model.fit(
    X_train=X_train.values,
    y_train=y_train,
    eval_set=[(X_test.values, y_test)],
    eval_metric=['accuracy'],
    max_epochs=50,         # Numero di epoche da configurare
    patience=10,           # Early stopping
    batch_size=1024,       # Dimensione del batch
    virtual_batch_size=128,  # Virtual batch size
    from_unsupervised=model,  # Indica che è supervisato
)

# Salva il modello ri-addestrato
model.save_model("tabnet_model9")
