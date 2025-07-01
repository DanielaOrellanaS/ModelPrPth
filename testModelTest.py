import pandas as pd
import torch
import torch.nn as nn
import pickle
import numpy as np

# =================== CONFIGURACIÓN ===================
SYMBOL = "GBPUSD"
MODEL_PATH = f"Trading_Model/trading_model_{SYMBOL}.pth"
PKL_PATH = f"Trading_Model/min_max_{SYMBOL}.pkl"
TEST_PATH = fr"C:\Users\user\OneDrive\Documentos\Trading\ModelPrPth\DataFiles\Data_{SYMBOL}_Test.xlsx"
OUTPUT_PATH = fr"C:\Users\user\OneDrive\Documentos\Trading\ModelPrPth\Predicciones\Resultados_Test_{SYMBOL}.xlsx"
MINIMO_GLOBAL = 0.0005  # umbral mínimo de profit

# =================== FUNCIONES ===================
def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def denormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def calcular_operacion(profit, minimo):
    if abs(profit) >= minimo:
        return 'BUY' if profit > 0 else 'SELL'
    else:
        return 'NO_SIGNAL'

# =================== MODELO ===================
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

# =================== CARGA MODELO Y MINMAX ===================
model = TradingModel()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

with open(PKL_PATH, "rb") as f:
    min_max = pickle.load(f)

# =================== CARGA Y PREPROCESAMIENTO TEST ===================
df = pd.read_excel(TEST_PATH)
df.columns = df.columns.str.strip()
df['fecha'] = df['fecha'].str.replace(',', '-', regex=False)
df['fecha'] = pd.to_datetime(df['fecha'], format="%Y-%m-%d %H:%M")

# Variables temporales
df['dia_semana'] = df['fecha'].dt.weekday / 6.0
df['hora'] = df['fecha'].dt.hour / 23.0
df['minuto'] = df['fecha'].dt.minute / 55.0

# Normalización precios 5min
for col in ['precioopen5', 'precioclose5', 'preciohigh5', 'preciolow5']:
    df[col] = normalize(df[col], min_max["min_precio5"], min_max["max_precio5"])
df['volume5'] = normalize(df['volume5'], min_max["min_volume5"], min_max["max_volume5"])

# Normalización precios 15min
for col in ['precioopen15', 'precioclose15', 'preciohigh15', 'preciolow15']:
    df[col] = normalize(df[col], min_max["min_precio15"], min_max["max_precio15"])
df['volume15'] = normalize(df['volume15'], min_max["min_volume15"], min_max["max_volume15"])

# Indicadores (RSI y Stochastic)
for col in ['rsi5', 'rsi15', 'iStochaMain5', 'iStochaSign5', 'iStochaMain15', 'iStochaSign15']:
    df[col] = df[col] / 100.0

# =================== PREDICCIÓN ===================
input_columns = [
    'dia_semana', 'hora', 'minuto',
    "precioopen5", "precioclose5", "precioclose5",
    "preciohigh5", "preciolow5", "volume5",
    "precioopen15", "precioclose15", "preciohigh15", "preciolow15", "volume15",
    "rsi5", "rsi15", "iStochaMain5", "iStochaSign5", "iStochaMain15", "iStochaSign15"
]

profits = []
tipos = []

for i in range(len(df)):
    entrada = df.loc[i, input_columns].values.astype(np.float32)
    entrada_tensor = torch.tensor(entrada).unsqueeze(0)

    with torch.no_grad():
        output = model(entrada_tensor).item()  # profit normalizado

    profit_real = denormalize(output, min_max["min_profit"], min_max["max_profit"])
    tipo = calcular_operacion(profit_real, MINIMO_GLOBAL)

    profits.append(profit_real)
    tipos.append(tipo)

    print(f"[{i+1}] {df.loc[i, 'fecha']} → profit: {profit_real:.6f} → tipo: {tipo}")

# =================== RESULTADOS ===================
df['profit_predicho'] = profits
df['tipo_predicho'] = tipos
df.to_excel(OUTPUT_PATH, index=False)

print("✅ Resultados guardados en:")
print(OUTPUT_PATH)
