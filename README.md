# Proyecto Trading - Modelos y API

Este proyecto contiene los modelos y utilidades para análisis y predicción de trading en pares de divisas como AUDUSD, EURUSD, GBPAUD y GBPUSD.

---

## 📂 Estructura principal

- **DataFiles/**: Archivos Excel con datos históricos para entrenamiento (`Data_Entrenamiento_PAR.xlsx`) y testeo (`Data_Test_PAR.xlsx`).
- **Predicciones/**: Resultados generados por los modelos tras el testeo.
- **Trading_Model/**: Modelos entrenados (`.pth`) y archivos con mínimos y máximos para normalización (`.pkl`).
- **BackupModels/**: Copias de seguridad de modelos entrenados.
- **env/**: Entorno virtual (no subir a repositorio).
- Scripts principales en la raíz para entrenamiento, testeo y uso de la API.

---

## 📄 Scripts principales

### 1. `convertUnityFilter.py`

Une y procesa dos archivos Excel (`TX_PAR` y `Transformer_PAR`) para preparar los datos completos que usa el modelo en entrenamiento y testeo.

- TX contiene los momentos de operación y profits.
- Transformer contiene indicadores técnicos y precios.
- El resultado es un archivo consolidado para análisis.

---

### 2. `model<PAR>.py` (e.g., `modelAUDUSD.py`)

Script para:

- Cargar datos de entrenamiento.
- Normalizar variables (precios, volúmenes, indicadores técnicos y tiempo).
- Entrenar un modelo MLP simple en PyTorch para predecir el profit.
- Guardar el modelo y los parámetros de normalización.

El mismo código se usa para los cuatro pares, cambiando solo el archivo de entrada y nombres.

---

### 3. `testModelForAll.py`

Permite testear cualquier modelo cargando los datos de test, aplicando la normalización y haciendo predicciones.

- Usa el modelo guardado y parámetros de normalización.
- Desnormaliza la predicción para obtener el profit real.
- Clasifica la señal (BUY, SELL, NO_SIGNAL).
- Guarda resultados en Excel.

---

### 4. `useApiWithExcel.py`

Envía datos desde un archivo Excel fila por fila al endpoint `/predict` de la API local para testear las predicciones en lote.

- Lee el archivo `Data_Test_<SYMBOL>.xlsx`.
- Envía los parámetros de indicadores y precios.
- Imprime el resultado de cada fila en consola.

---

### 5. `useApiWithUrl.py`

Automatiza el testeo enviando peticiones GET desde URLs almacenadas en un Excel.

- Cambia la URL del dominio remoto a localhost para pruebas.
- Ejecuta todas las peticiones y guarda las respuestas en el archivo.
- Útil para validar grandes volúmenes de datos desde URLs predefinidas.

---

## ⚙️ Configuración y uso

- Cambiar variables globales (`SYMBOL`, paths de archivos) según el par y entorno.
- Entrenar modelos con `model<PAR>.py`.
- Ejecutar test con `testModelForAll.py`.
- Probar API local con `useApiWithExcel.py` o `useApiWithUrl.py`.
- Los archivos `.pth` y `.pkl` se generan automáticamente y se usan para inferencia.

---

## Dependencias

- Python 3.11.3
- pandas
- numpy
- torch
- openpyxl (para Excel)
- requests (para test API)

Instalar con:

```bash
pip install -r requirements.txt
