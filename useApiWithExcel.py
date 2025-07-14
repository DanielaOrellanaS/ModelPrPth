import pandas as pd
import requests
from datetime import datetime

# Ruta del archivo Excel con tus datos
SYMBOL = "GBPAUD"  
excel_path = fr'C:\Users\user\OneDrive\Documentos\Trading\ModelPrPth\ModelAndTest\DataFiles\Data_Test_{SYMBOL}.xlsx'
# Cargar datos
df = pd.read_excel(excel_path)
df.columns = df.columns.str.strip()

# Reemplazar comas por guiones en fecha si vienen con coma
df['fecha'] = df['fecha'].str.replace(',', '-', regex=False)

# Convertir fecha a formato ISO
df['fecha'] = pd.to_datetime(df['fecha'], format="%Y-%m-%d %H:%M")
df['fecha'] = df['fecha'].dt.strftime("%Y-%m-%dT%H:%M")  # Formato ISO para URL

# URL base del endpoint
base_url = "http://localhost:8000/predict"

# Recorrer filas
for i, row in df.iterrows():
    params = {
        "symbol": row["simbolo"],
        "fecha": row["fecha"],
        "o5": row["precioopen5"],
        "c5": row["precioclose5"],
        "h5": row["preciohigh5"],
        "l5": row["preciolow5"],
        "v5": row["volume5"],
        "o15": row["precioopen15"],
        "c15": row["precioclose15"],
        "h15": row["preciohigh15"],
        "l15": row["preciolow15"],
        "v15": row["volume15"],
        "r5": row["rsi5"],
        "r15": row["rsi15"],
        "m5": row["iStochaMain5"],
        "s5": row["iStochaSign5"],
        "m15": row["iStochaMain15"],
        "s15": row["iStochaSign15"],
    }


    try:
        response = requests.get(base_url, params=params)
        result = response.json()
        
        print(f"🕒 Fila {i+1} | Fecha: {params['fecha']}")
        if "error" in result:
            print("❌ Error:", result["error"])
        else:
            print(f"✅ Profit: {result['valor_profit']:.6f} | Tipo operación: {result['RESULTADO']}")
        print("-" * 60)
    except Exception as e:
        print(f"❌ Error en la fila {i+1}: {e}")
