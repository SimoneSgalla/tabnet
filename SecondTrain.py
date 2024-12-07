from pytorch_tabnet.tab_model import TabNetClassifier
import pandas as pd
import numpy as np

# Carica il modello salvato
model = TabNetClassifier()
model.load_model("tabnet_model2.zip")

# Carica e prepara il nuovo dataset
new_data = pd.read_csv("../Dataset1/Friday-16-02-2018_TrafficForML_CICFlowMeter.csv")
new_data = new_data.replace([np.inf, -np.inf], np.nan)
new_data = new_data.fillna(0)  # Sostituisci con la strategia di preprocessamento usata in precedenza

# Separa le feature (X) e le label (y)
new_data = new_data.drop(columns=["Timestamp"])
X_new = new_data.drop(columns=["Label"])
y_new = new_data["Label"]

# Converte le etichette in numeri se necessario (ad esempio, categorizzazione)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_new = encoder.fit_transform(y_new)

# Ri-Addestra il modello
model.fit(
    X_train=X_new.values,  # Assicurati di convertire in NumPy array
    y_train=y_new,
    max_epochs=50,         # Numero di epoche da configurare
    patience=10,           # Early stopping
    batch_size=1024,       # Dimensione del batch
    virtual_batch_size=128,  # Virtual batch size
    from_unsupervised=model # Indica che è supervisato
)

# Salva il modello ri-addestrato
model.save_model("tabnet_model3.zip")
