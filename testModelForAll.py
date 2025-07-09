import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np

# ============== Parámetro global ==============
SYMBOL = "GBPUSD"  

# ============== Configuración de rutas ==============
MINIMO_GLOBAL = 0.0005
model_path = f"Trading_Model/trading_model_{SYMBOL}.pth"
minmax_path = f"Trading_Model/min_max_{SYMBOL}.pkl"
test_path = fr'C:\Users\user\OneDrive\Documentos\Trading\ModelPrPth\ModelAndTest\DataFiles\Data_Test_{SYMBOL}.xlsx'
output_path = fr'C:\Users\user\OneDrive\Documentos\Trading\ModelPrPth\ModelAndTest\Predicciones\Data_Profit_Salida_{SYMBOL}.xlsx'

# ============== Funciones utilitarias ==============
def normalize(column, min_val, max_val):
    return (column - min_val) / (max_val - min_val)

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

def calcular_operacion(profit, minimo):
    if abs(profit) > minimo:
        return 'BUY' if profit > 0 else 'SELL'
    else:
        return 'NO_SIGNAL'

# ============== Cargar modelo ==============
class TradingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = TradingModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# ============== Cargar min/max ==============
with open(minmax_path, "rb") as f:
    min_max = pickle.load(f)

min_profit = min_max["min_profit"]
max_profit = min_max["max_profit"]
min_precio5 = min_max["min_precio5"]
max_precio5 = min_max["max_precio5"]
min_precio15 = min_max["min_precio15"]
max_precio15 = min_max["max_precio15"]

# ============== Cargar y procesar datos ==============
df = pd.read_excel(test_path)
df.columns = df.columns.str.strip()
df['fecha'] = df['fecha'].str.replace(',', '-', regex=False)
df['fecha'] = pd.to_datetime(df['fecha'], format="%Y-%m-%d %H:%M")

df['dia_semana'] = df['fecha'].dt.weekday / 6.0
df['hora'] = df['fecha'].dt.hour / 23.0
df['minuto'] = df['fecha'].dt.minute / 55.0

# Normalización precios 5min
for col in ['precioopen5', 'precioclose5', 'preciohigh5', 'preciolow5']:
    df[col] = normalize(df[col], min_precio5, max_precio5)
df['volume5'] = normalize(df['volume5'], df['volume5'].min(), df['volume5'].max())

# Normalización precios 15min
for col in ['precioopen15', 'precioclose15', 'preciohigh15', 'preciolow15']:
    df[col] = normalize(df[col], min_precio15, max_precio15)
df['volume15'] = normalize(df['volume15'], df['volume15'].min(), df['volume15'].max())

# Normalización indicadores
for col in ['rsi5', 'rsi15', 'iStochaMain5', 'iStochaSign5', 'iStochaMain15', 'iStochaSign15']:
    df[col] = df[col] / 100.0

input_columns = [
    'dia_semana', 'hora', 'minuto',
    "precioopen5", "precioclose5", "precioclose5",
    "preciohigh5", "preciolow5", "volume5",
    "precioopen15", "precioclose15", "preciohigh15", "preciolow15", "volume15",
    "rsi5", "rsi15", "iStochaMain5", "iStochaSign5", "iStochaMain15", "iStochaSign15"
]

# ============== Predicción ==============
profit_pred = []
tipo_pred = []
profit_raw = []

print(f"🔎 Procesando test para símbolo: {SYMBOL}")
for i in range(len(df)):
    input_vals = df.loc[i, input_columns].values.astype(np.float32)
    input_tensor = torch.tensor(input_vals).unsqueeze(0)

    with torch.no_grad():
        raw_output = model(input_tensor).item()
    profit_raw.append(raw_output)
    profit = denormalize(raw_output, min_profit, max_profit)
    tipo = calcular_operacion(profit, MINIMO_GLOBAL)

    profit_pred.append(profit)
    tipo_pred.append(tipo)

    print(f"Registro {i+1:02d} | Fecha: {df.loc[i, 'fecha']}")
    print(f"  Input normalizado: {input_vals}")
    print(f"  Output raw: {raw_output:.6f}")
    print(f"  Profit (desnormalizado): {profit:.6f}")
    print(f"  Tipo de operación: {tipo}")
    print("-" * 50)

# ============== Guardar resultados ==============
df['profit_normalizado'] = profit_raw
df['profit_desnormalizado'] = profit_pred
df['tipo_prediction'] = tipo_pred

# Eliminar columnas que no deben salir en el Excel
df.drop(columns=['dia_semana', 'hora', 'minuto'], inplace=True)

df.to_excel(output_path, index=False)

print(f"✅ Resultados guardados en:\n{output_path}")
print("min_max usados:", min_max)
