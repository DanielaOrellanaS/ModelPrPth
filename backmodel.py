import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import os 

# Función de normalización
def normalize(column, min_val, max_val):
    return (column - min_val) / (max_val - min_val)

# Cargar datos de entrenamiento
file_path = r'C:\Users\user\OneDrive\Documentos\Trading\ModelPth\DataFiles\Data_AUDUSD_2025-05-09.xlsx'
data = pd.read_excel(file_path)
data.columns = data.columns.str.strip()

# Convertir la columna `tipo` a valores binarios
def tipo_to_binary(tipo):
    if tipo == 'BUY':
        return 0.9  # Aproximado a 1
    elif tipo == 'SELL':
        return -0.9  # Aproximado a -1
    else:
        return 0  # Mantiene 0 para "ninguno"

data['tipo'] = data['tipo'].apply(tipo_to_binary)

# Aumentar dif 
data['dif'] = abs(data['preciohigh5'] - data['preciolow5'])
data.loc[data['dif'] <= 0.001, 'tipo_prediction'] = 0

print(data[['preciohigh5', 'preciolow5', 'dif']].head())

# Normalización de las columnas de precios para 5
def normalize(column, min_val, max_val):
    return (column - min_val) / (max_val - min_val)

min_precio_global_5 = data['preciolow5'].min()
max_precio_global_5 = data['preciohigh5'].max()

print("PRECIOS LOW: ", min_precio_global_5)
print("PRECIOS HIGH: ", max_precio_global_5)


data['precioopen5'] = normalize(data['precioopen5'], min_precio_global_5, max_precio_global_5)
data['precioclose5'] = normalize(data['precioclose5'], min_precio_global_5, max_precio_global_5)
data['preciohigh5'] = normalize(data['preciohigh5'], min_precio_global_5, max_precio_global_5)
data['preciolow5'] = normalize(data['preciolow5'], min_precio_global_5, max_precio_global_5)

# Normalización de la columna 'volume5'
min_volume_global_5 = data['volume5'].min()
max_volume_global_5 = data['volume5'].max()

data['volume5'] = normalize(data['volume5'], min_volume_global_5, max_volume_global_5)

# Normalización de las columnas de precios para 15
min_precio_global_15 = data['preciolow15'].min()
max_precio_global_15 = data['preciohigh15'].max()

data['precioopen15'] = normalize(data['precioopen15'], min_precio_global_15, max_precio_global_15)
data['precioclose15'] = normalize(data['precioclose15'], min_precio_global_15, max_precio_global_15)
data['preciohigh15'] = normalize(data['preciohigh15'], min_precio_global_15, max_precio_global_15)
data['preciolow15'] = normalize(data['preciolow15'], min_precio_global_15, max_precio_global_15)

# Normalización de la columna 'volume15'
min_volume_global_15 = data['volume15'].min()
max_volume_global_15 = data['volume15'].max()

data['volume15'] = normalize(data['volume15'], min_volume_global_15, max_volume_global_15)

# Normalización de RSI (dividiendo por 100)
data['rsi5'] = data['rsi5'] / 100
data['rsi15'] = data['rsi15'] / 100

# Normalización de iStochaMain e iStochaSign (dividiendo por 100)
data['iStochaMain5'] = data['iStochaMain5'] / 100
data['iStochaSign5'] = data['iStochaSign5'] / 100
data['iStochaMain15'] = data['iStochaMain15'] / 100
data['iStochaSign15'] = data['iStochaSign15'] / 100

# Normalización de pr1 y pr2 usando los valores mínimo y máximo de pr1
min_pr1_global = data['pr1'].min()
max_pr1_global = data['pr1'].max()

data['pr1'] = normalize(data['pr1'], min_pr1_global, max_pr1_global)
data['pr2'] = normalize(data['pr2'], min_pr1_global, max_pr1_global)

# Normalización de 'profit'
data['profit'] = normalize(data['profit'], min_pr1_global, max_pr1_global)

# Cálculo del mínimo y máximo de 'periodos'
min_periodos = data['periodos'].min()
max_periodos = data['periodos'].max()

# Normalización de 'periodos'
data['periodos'] = normalize(data['periodos'], min_periodos, max_periodos)

# Ordenar los datos por fecha
data = data.sort_values(by='fecha', ascending=True)

# Convertir la columna 'fecha' a datetime si no lo está
data['fecha'] = pd.to_datetime(data['fecha'], format="%Y,%m,%d %H:%M")

# Extraer la hora en formato decimal (ej. 13:30 -> 13.5)
data['hora_decimal'] = data['fecha'].dt.hour + data['fecha'].dt.minute / 60

# Normalizar la hora
data['hora_normalizada'] = data['hora_decimal'] / 24  

# Filtrar para quedarte solo con horas válidas si quieres (opcional)
data = data[(data['hora_decimal'] >= 9) & (data['hora_decimal'] <= 16)]

# Seleccionar características y variable objetivo
input_columns = [
    "precioopen5", "precioclose5", "preciohigh5", "preciolow5", "volume5",
    "precioopen15", "precioclose15", "preciohigh15", "preciolow15", "volume15",
    "rsi5", "rsi15", "iStochaMain5", "iStochaSign5", "iStochaMain15", "iStochaSign15",
    "fill", "dif", "hora_normalizada"
]
target_column = "tipo"

# Definir conjunto de datos en PyTorch
class TradingDataset(Dataset):
    def __init__(self, df):
        self.features = df[input_columns].values
        self.labels = df[target_column].values.reshape(-1, 1)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# Crear DataLoader
dataset = TradingDataset(data)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Definir modelo en PyTorch
class TradingModelAUDUSD(nn.Module):
    def __init__(self):
        super(TradingModelAUDUSD, self).__init__()
        self.fc1 = nn.Linear(len(input_columns), 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# Entrenamiento del modelo
model = TradingModelAUDUSD()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 50
for epoch in range(epochs):
    for features, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}')

# Guardar el modelo entrenado
models_dir = 'Models'
os.makedirs(models_dir, exist_ok=True)
model_path = os.path.join(models_dir, 'trading_model_AUDUSD.pth')
model_path = os.path.join(models_dir, 'trading_model_AUDUSD_with_hour.pth')
torch.save(model.state_dict(), model_path)

# Cargar y normalizar los datos de prueba
file_test = r'C:\Users\user\OneDrive\Documentos\Trading\ModelPth\DataFiles\Data_AUDUSD_2025-04-23.xlsx'
test_data = pd.read_excel(file_test)

# Aumentar dif 
test_data['dif'] = abs(test_data['preciohigh5'] - test_data['preciolow5'])

# Convertir la columna 'fecha' a datetime
test_data['fecha'] = pd.to_datetime(test_data['fecha'], format="%Y,%m,%d %H:%M")

# Calcular hora decimal y normalizar
test_data['hora_decimal'] = test_data['fecha'].dt.hour + test_data['fecha'].dt.minute / 60
test_data['hora_normalizada'] = test_data['hora_decimal'] / 24

# Filtrar para quedarte solo con datos entre 9 y 16 (opcional)
test_data = test_data[(test_data['hora_decimal'] >= 9) & (test_data['hora_decimal'] <= 16)]

# Crear un diccionario con los valores min y max utilizados en la normalización
min_max_dict = {
    "precioopen5": (min_precio_global_5, max_precio_global_5),
    "precioclose5": (min_precio_global_5, max_precio_global_5),
    "preciohigh5": (min_precio_global_5, max_precio_global_5),
    "preciolow5": (min_precio_global_5, max_precio_global_5),
    "volume5": (min_volume_global_5, max_volume_global_5),
    
    "precioopen15": (min_precio_global_15, max_precio_global_15),
    "precioclose15": (min_precio_global_15, max_precio_global_15),
    "preciohigh15": (min_precio_global_15, max_precio_global_15),
    "preciolow15": (min_precio_global_15, max_precio_global_15),
    "volume15": (min_volume_global_15, max_volume_global_15),
    
    "rsi5": (0, 100),
    "rsi15": (0, 100),
    "iStochaMain5": (0, 100),
    "iStochaSign5": (0, 100),
    "iStochaMain15": (0, 100),
    "iStochaSign15": (0, 100),
    
    "pr1": (min_pr1_global, max_pr1_global),
    "pr2": (min_pr1_global, max_pr1_global),
    "profit": (min_pr1_global, max_pr1_global),
    
    "periodos": (min_periodos, max_periodos)
}

# Aplicar la misma normalización a los datos de prueba
for col in input_columns:
    if col in min_max_dict:
        min_val, max_val = min_max_dict[col]
        test_data[col] = normalize(test_data[col], min_val, max_val)

test_features = test_data[input_columns].values
test_features = torch.tensor(test_features, dtype=torch.float32)

# Cargar el modelo y predecir
test_model = TradingModelAUDUSD()
test_model.load_state_dict(torch.load(model_path, weights_only=True))
test_model.eval()
predictions = test_model(test_features).detach().numpy()
test_data['tipo_prediction'] = predictions
test_data.loc[test_data['dif'] <= 0.001, 'tipo_prediction'] = 0

# Ordenar fecha del mas antiguo al mas reciente
test_data["fecha"] = pd.to_datetime(test_data["fecha"], format="%Y,%m,%d %H:%M")
test_data = test_data.sort_values(by='fecha', ascending=True)
test_data["fecha"] = test_data["fecha"].dt.strftime("%Y,%m,%d %H:%M")

# Guardar el DataFrame actualizado en un nuevo archivo Excel
output_file = r'C:\Users\user\OneDrive\Documentos\Trading\ModelPth\predicciones_AUDUSD.xlsx'
test_data.to_excel(output_file, index=False)