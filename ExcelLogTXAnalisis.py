import pandas as pd
import os

# ========= Parámetros =========
SIMBOLO = "GBPAUD"
FECHA = "2025-06-26"
BASE_DIR = r"C:\Users\user\Downloads"
BASE_OUT = r"C:\Users\user\OneDrive\Documentos\Trading\ModelPrPth"
CARPETA_SALIDA = os.path.join(BASE_OUT, "AnalisisDatosTxLog")
os.makedirs(CARPETA_SALIDA, exist_ok=True)

# ========= Archivos =========
ruta_tx = os.path.join(BASE_DIR, f"TX_{SIMBOLO}_{FECHA}.csv")
ruta_log = os.path.join(BASE_DIR, f"log_url_{SIMBOLO}_{FECHA.replace('-', '.')}.csv")

# ========= Cargar archivos =========
df_tx = pd.read_csv(ruta_tx, sep=';')
df_log = pd.read_csv(ruta_log, sep=';')

# ========= Normalizar columnas =========
df_tx.columns = [c.strip().lower() for c in df_tx.columns]
df_log.columns = [c.strip().lower() for c in df_log.columns]

# ========= Convertir fechas a datetime y clave texto =========
df_tx['fecha'] = pd.to_datetime(df_tx['fecha'], errors='coerce')
df_log['date'] = pd.to_datetime(df_log['date'], format='%Y.%m.%d %H:%M', errors='coerce')

df_tx['clave_merge'] = df_tx['fecha'].dt.strftime('%Y-%m-%d %H:%M')
df_log['clave_merge'] = df_log['date'].dt.strftime('%Y-%m-%d %H:%M')

# ========= Diccionario de TX para emparejar =========
tx_dict = df_tx.set_index('clave_merge')[['pr1', 'pr2', 'dif', 'tipo']].to_dict('index')

# ========= Asignar TX a log, solo si hay coincidencia exacta =========
datos_pr1, datos_pr2, datos_dif, datos_tipo = [], [], [], []

usados = set()
for clave in df_log['clave_merge']:
    if clave in tx_dict and clave not in usados:
        row_tx = tx_dict[clave]
        datos_pr1.append(row_tx['pr1'])
        datos_pr2.append(row_tx['pr2'])
        datos_dif.append(row_tx['dif'])
        datos_tipo.append(row_tx['tipo'])
        usados.add(clave)
    else:
        datos_pr1.append(None)
        datos_pr2.append(None)
        datos_dif.append(None)
        datos_tipo.append(None)

df_log['pr1'] = datos_pr1
df_log['pr2'] = datos_pr2
df_log['dif'] = datos_dif
df_log['tipo'] = datos_tipo

# ========= Encontrar TX no usados (que no estaban en log) =========
tx_sin_usar = df_tx[~df_tx['clave_merge'].isin(usados)].copy()
tx_sin_usar['date'] = tx_sin_usar['fecha']
tx_sin_usar['clave_merge'] = tx_sin_usar['fecha'].dt.strftime('%Y-%m-%d %H:%M')

# Crear columnas vacías para los campos que están en log pero no en TX
for col in df_log.columns:
    if col not in tx_sin_usar.columns:
        tx_sin_usar[col] = None

# ========= Unir ambos y ordenar cronológicamente =========
df_final = pd.concat([df_log, tx_sin_usar[df_log.columns]], ignore_index=True)
df_final = df_final.sort_values(by='date').drop(columns=['clave_merge'])

# ========= Guardar Excel =========
ruta_salida = os.path.join(CARPETA_SALIDA, f"Analisis_{SIMBOLO}.xlsx")
df_final.to_excel(ruta_salida, index=False)

print("✅ Archivo final generado en:", ruta_salida)
