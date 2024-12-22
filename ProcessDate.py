import os
import pandas as pd

# Percorso della directory e file di output
directory_path = '../Dataset1'
output_file = 'Csv/TransTimestamp.csv'

# Leggi tutti i file CSV nella directory specificata
all_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.csv')]

# Controlla se ci sono file CSV
if not all_files:
    raise FileNotFoundError(f"Nessun file CSV trovato nella directory {directory_path}")

# Carica tutti i file e concatena i dati
data = []
for file in all_files:
    data.append(pd.read_csv(file, usecols=['Timestamp']))

# Combina tutti i DataFrame in un unico DataFrame
df = pd.concat(data, ignore_index=True)

# Verifica se la colonna 'Timestamp' esiste
if 'Timestamp' not in df.columns:
    raise ValueError("La colonna 'Timestamp' non è presente nei file CSV")

# Converte la colonna 'Timestamp' in formato datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

# Rimuovi eventuali valori non validi
df = df.dropna(subset=['Timestamp'])

# Estrai giorno della settimana, ora e minuto
df['dow'] = df['Timestamp'].dt.day_name()
df['hour'] = df['Timestamp'].dt.hour
df['minute'] = df['Timestamp'].dt.minute

# Rimuovi la colonna originale 'Timestamp'
df = df.drop(columns=['Timestamp'])

# Salva il DataFrame risultante in un nuovo file CSV
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df.to_csv(output_file, index=False)

# Stampa informazioni sulle colonne uniche
print(f"Unique values: dow={df['dow'].nunique()}, hour={df['hour'].nunique()}, minute={df['minute'].nunique()}")
