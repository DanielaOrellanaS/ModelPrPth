import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import urllib.parse as up
import pickle
import requests


# === Configura tu símbolo y archivo ===
symbol = "AUDUSD"  # Cambia este valor según el par
fecha_archivo = "2025.07.07"
xlsx_path = fr"C:\Users\user\Downloads\Salidas\test_log_url_{symbol}_{fecha_archivo}.xlsx"

# === Leer archivo Excel ===
df = pd.read_excel(xlsx_path)
df.columns = df.columns.str.strip()

# === Reemplazar dominio Render por localhost ===
df['url_local'] = df['post'].str.replace("https://apiprpth.onrender.com", "http://localhost:8000", regex=False)

# === Inicializar columnas de salida ===
df["prediccion_modelo"] = ""
df["tipo_modelo"] = ""

# === Enviar peticiones ===
for i, row in df.iterrows():
    url = row["url_local"]
    try:
        response = requests.get(url)
        result = response.json()

        print(f"Fila {i+1} result: {result}")  # <- para ver qué devuelve en cada caso
        
        if "error" in result:
            df.at[i, "prediccion_modelo"] = "ERROR"
            df.at[i, "tipo_modelo"] = result["error"]
        else:
            df.at[i, "prediccion_modelo"] = result["valor_profit"]
            df.at[i, "tipo_modelo"] = result["RESULTADO"]
            print(f"✅ Fila {i+1} OK - Profit: {result['valor_profit']:.6f} - Tipo: {result['RESULTADO']}")
    except Exception as e:
        df.at[i, "prediccion_modelo"] = "ERROR"
        df.at[i, "tipo_modelo"] = str(e)
        print(f"❌ Fila {i+1} Error: {e}")


# === Guardar de nuevo el archivo Excel con las nuevas columnas ===
df.to_excel(xlsx_path, index=False)
print(f"\n✅ Archivo Excel actualizado con predicciones: {xlsx_path}")