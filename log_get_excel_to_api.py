import pandas as pd
import requests
import os

# === Ruta del archivo CSV de entrada ===
csv_path = r"C:\Users\user\Downloads\log_url_GBPUSD_2025.06.25.csv"

# === Leer CSV usando punto y coma como separador ===
df_csv = pd.read_csv(csv_path, sep=';')

# === Convertir a Excel (temporal) para depurar mejor ===
excel_temp_path = csv_path.replace(".csv", ".xlsx")
df_csv.to_excel(excel_temp_path, index=False)
print(f"📄 Archivo CSV convertido a Excel temporal: {excel_temp_path}")

# === Leer el archivo Excel ===
df = pd.read_excel(excel_temp_path)
df.columns = df.columns.str.strip()  # limpiar nombres de columnas

# === Verificar columna 'post' ===
if 'post' not in df.columns or 'version' not in df.columns:
    raise ValueError(f"Falta columna requerida. Columnas disponibles: {df.columns.tolist()}")

# === Resultado final ===
resultados = []

# === Recorrer filas ===
for i, row in df.iterrows():
    post_url = str(row['post'])
    version = str(row.get('version')).strip()

    if version != 'v2':
        print(f"⏭️ ({i+1}) Fila ignorada, versión no es v2.")
        continue

    if '/predict?' not in post_url or 'fecha=' not in post_url:
        print(f"⏭️ ({i+1}) Fila ignorada, post incompleto.")
        continue

    # Extraer solo la parte desde /predict? y construir URL local completa
    query = post_url.split("/predict?")[-1]
    local_url = f"http://127.0.0.1:8000/predict?{query}"

    print(f"\n🔄 ({i+1}/{len(df)}) Petición:\n{local_url}")

    try:
        response = requests.get(local_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            valor = data.get("valor_profit")
            resultado = data.get("RESULTADO")
            print(f"✅ Resultado: {resultado} | Profit: {valor}")
            resultado_dict = {
                "valor_profit": valor,
                "RESULTADO": resultado
            }
        else:
            print(f"❌ Error HTTP {response.status_code}")
            resultado_dict = {
                "valor_profit": None,
                "RESULTADO": f"Error {response.status_code}"
            }
    except Exception as e:
        print(f"❌ Excepción: {e}")
        resultado_dict = {
            "valor_profit": None,
            "RESULTADO": f"Excepción: {str(e)}"
        }

    # Agregar al resultado final
    resultados.append({
        "date": row.get("date"),
        "version": version,
        "simb": row.get("simb"),
        "timeframe": row.get("timeframe"),
        "post": local_url,
        **resultado_dict
    })

# === Guardar resultados en nuevo archivo Excel ===
output_path = r"C:\Users\user\Downloads\respuestas_local_GBPUSD.xlsx"
pd.DataFrame(resultados).to_excel(output_path, index=False)

print(f"\n✅ Archivo final guardado en: {output_path}")
