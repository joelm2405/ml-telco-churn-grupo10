# Proyecto 1 — ML (CS3061): Clasificación Telco Churn

Proyecto de Machine Learning para predecir **Customer Churn** en clientes de telecomunicaciones.

El problema se plantea como una **clasificación binaria**, donde:

- `Churn = Yes` indica que el cliente abandona la empresa.
- `Churn = No` indica que el cliente permanece.

Se trabajó con los archivos oficiales del reto:

- `train.csv`: contiene las variables predictoras y la variable objetivo `Churn`.
- `test.csv`: contiene las mismas variables predictoras, pero sin `Churn`; se usa solo para generar predicciones finales.

El dataset presenta desbalance de clases: aproximadamente **73.4% `No Churn`** y **26.6% `Churn`**.

---

## Estructura del repositorio

```text
.
├── figures/
│   ├── 01_churn_balance.png
│   ├── 02_churn_by_contract.png
│   ├── 06_threshold_optim.png
│   └── 08_confusion_optim-040.png
│
├── notebooks/
│   └── Proyecto_ML_Churn.ipynb
│
├── train.csv
├── test.csv
├── submission_kaggle_f1.csv
├── submission_threshold_costo_minimo.csv
├── submission_threshold_operativo.csv
├── main.tex
├── PROYECTO_MACHINE_LEARNING___GRUPO_10.pdf
├── requirements.txt
└── README.md
```

> Si el notebook no está dentro de `notebooks/`, sino en la raíz, reemplazar `notebooks/Proyecto_ML_Churn.ipynb` por `Proyecto_ML_Churn.ipynb`.

---

## Pipeline del proyecto

El notebook implementa el siguiente flujo:

1. **Carga de datos**
   - Lectura de `train.csv` y `test.csv`.
   - Separación de `customerID` para conservarlo en los archivos de predicción.

2. **Preprocesamiento**
   - Eliminación de `customerID` como variable predictora.
   - Conversión de `TotalCharges` a numérico.
   - Imputación de valores faltantes con la mediana.
   - Reemplazo de categorías redundantes:
     - `No internet service` → `No`
     - `No phone service` → `No`

3. **Ingeniería de variables**
   - `tenure_bucket`
   - `avg_charges_per_month`
   - `num_services`
   - `is_new_customer`

4. **Encoding**
   - Variables binarias convertidas a 0/1.
   - Variables multiclase transformadas con One-Hot Encoding.

5. **División interna**
   - `train.csv` se divide en:
     - 80% entrenamiento
     - 20% validación
   - Split estratificado para mantener la proporción de clases.

6. **Escalado numérico**
   - Se aplica `StandardScaler` a variables numéricas.
   - El scaler se ajusta solo con entrenamiento para evitar data leakage.

7. **Modelado**
   - Regresión Logística con regularización L2.
   - Random Forest.
   - Optimización con `GridSearchCV`.
   - Validación cruzada estratificada con 5 folds.
   - Métrica de optimización: F1-score.

8. **Evaluación**
   - Accuracy.
   - Precision.
   - Recall.
   - F1-score.
   - Matriz de confusión.
   - Matriz de costos.
   - Comparación de thresholds.

9. **Predicciones finales**
   - Submission optimizada por F1-score para Kaggle/Cadem.
   - Submission con threshold de costo mínimo.
   - Submission con threshold operativo costo-beneficio.

---

## Resultados clave

### Métricas con threshold base 0.5

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Regresión Logística | 0.7560 | 0.5261 | 0.8094 | 0.6377 |
| Random Forest | 0.7728 | 0.5544 | 0.7324 | 0.6311 |

Con threshold 0.5, **Random Forest obtiene mayor accuracy**, mientras que **Regresión Logística obtiene mayor recall** para la clase `Churn`.

Esto significa que Random Forest acierta más en general, pero Regresión Logística detecta una mayor proporción de clientes que realmente abandonan la empresa.

---

### Mejor configuración para Kaggle/Cadem

La métrica oficial de Kaggle/Cadem es **F1-score**, por lo que se seleccionó el modelo y threshold que maximizan F1 en validación.

| Criterio | Modelo | Threshold | F1 validación | Accuracy validación |
|---|---|---:|---:|---:|
| Kaggle/Cadem | Regresión Logística | 0.55 | 0.6496 | 0.7808 |

Archivo recomendado para subir:

```text
submission_kaggle_f1.csv
```

Distribución de predicciones sobre `test.csv`:

| Clase predicha | Cantidad |
|---|---:|
| No | 905 |
| Yes | 504 |

---

### Mejor configuración por costo mínimo

Para el análisis costo-beneficio, el menor costo esperado se obtuvo con **Random Forest** y threshold **0.12**.

| Modelo | Threshold | Costo con t=0.5 | Costo con t* | Ahorro |
|---|---:|---:|---:|---:|
| Random Forest | 0.12 | $46,470 | $27,550 | 40.7% |

Este threshold minimiza el costo esperado porque reduce casi al mínimo los falsos negativos. Sin embargo, genera una cantidad elevada de falsos positivos, por lo que se interpreta como un **mínimo costo teórico**, no necesariamente como la opción más práctica para una campaña real.

---

### Alternativa costo-beneficio operativa

La alternativa operativa elegida fue **Regresión Logística con threshold 0.40**, porque reduce falsos positivos y mantiene un recall alto.

| Modelo | Threshold | Accuracy | Precision | Recall | F1 | Costo |
|---|---:|---:|---:|---:|---:|---:|
| Regresión Logística | 0.40 | 0.6992 | 0.4654 | 0.8997 | 0.6135 | $31,630 |

Esta configuración no minimiza el costo absoluto, pero ofrece un balance más razonable entre:

- Detectar clientes que realmente podrían hacer churn.
- Reducir falsas alarmas.
- Mantener un costo menor que el threshold base de 0.5.

---

## Análisis de thresholds

Se analizaron distintos thresholds porque el modelo primero produce probabilidades y luego estas se convierten en clases (`Churn` / `No Churn`) usando un umbral de decisión.

| Criterio | Modelo | Threshold | F1 | Costo |
|---|---|---:|---:|---:|
| Costo mínimo teórico | Random Forest | 0.12 | 0.5113 | $27,550 |
| Costo-beneficio operativo | Regresión Logística | 0.40 | 0.6135 | $31,630 |
| Baseline | Random Forest | 0.50 | 0.6311 | $46,470 |
| Kaggle/Cadem F1 | Regresión Logística | 0.55 | 0.6496 | $42,990 |

El threshold **0.12** minimiza el costo esperado, pero produce demasiados falsos positivos.  
El threshold **0.40** representa una alternativa más equilibrada desde una perspectiva operativa.  
El threshold **0.55** se usa para la submission oficial porque maximiza F1-score en validación.

---

## Matriz de costos

Se utilizó la siguiente matriz de costos para el análisis costo-beneficio:

| Real / Predicción | No Churn | Churn |
|---|---:|---:|
| No Churn | $0 | $20 |
| Churn | $400 | $50 |

El costo total esperado se calcula como:

```text
Costo = TN*0 + FP*20 + FN*400 + TP*50
```

Donde:

- **TN**: cliente no se iba y el modelo predijo `No Churn`.
- **FP**: cliente no se iba, pero el modelo predijo `Churn`.
- **FN**: cliente sí se iba, pero el modelo predijo `No Churn`.
- **TP**: cliente sí se iba y el modelo predijo `Churn`.

El falso negativo tiene el costo más alto porque representa un cliente que realmente se iba, pero el modelo no lo detectó. En cambio, un falso positivo implica una posible acción de retención innecesaria, pero su costo es menor.

---

## Submission oficial

Para Kaggle/Cadem se prioriza F1-score, porque es la métrica oficial de evaluación.

La mejor configuración para submission fue:

```text
Modelo: Regresión Logística
Threshold: 0.55
F1 validación: 0.6496
Accuracy validación: 0.7808
```

Archivo generado:

```text
submission_kaggle_f1.csv
```

Distribución de predicciones en `test.csv`:

| Clase predicha | Cantidad |
|---|---:|
| No | 905 |
| Yes | 504 |

> Nota: si la plataforma solicita formato binario, se debe convertir `Yes → 1` y `No → 0`.

---

## Ejecutar el proyecto

### Opción 1: Jupyter Notebook

Abrir y ejecutar:

```text
notebooks/Proyecto_ML_Churn.ipynb
```

Ejecutar las celdas en orden desde el inicio.

Si el notebook está en la raíz del repositorio, abrir:

```text
Proyecto_ML_Churn.ipynb
```

---

### Opción 2: Entorno local

Crear entorno virtual:

```bash
python -m venv venv
```

Activar entorno:

```bash
# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Compilar el informe

El informe está en formato LaTeX en la raíz del repositorio:

```text
main.tex
```

Para compilar localmente:

```bash
pdflatex main.tex
pdflatex main.tex
```

También se puede compilar en Overleaf subiendo:

```text
main.tex
figures/
```

El PDF final generado es:

```text
PROYECTO_MACHINE_LEARNING___GRUPO_10.pdf
```

---

## Reproducibilidad

Se utilizó:

```python
SEED = 42
```

La semilla se aplica en:

- División train/validación.
- Validación cruzada.
- Modelos con componente aleatorio.
- Random Forest.

Además, el escalado numérico se ajusta solo con el conjunto de entrenamiento para evitar data leakage.

---

## Integrantes

| Integrante | Contribución | % |
|---|---|---:|
| Alejandro Marcelo | Limpieza de datos, tratamiento de valores faltantes, codificación de variables y escalado numérico | 25% |
| Ferrel Valentino Infante Garcia | Implementación de modelos, validación cruzada, GridSearchCV y comparación de Regresión Logística con Random Forest | 25% |
| Leonardo Martinez Aquino | Análisis exploratorio de datos, interpretación de gráficos, análisis del desbalance de clases y selección de variables relevantes | 25% |
| Joel David Miguel Fernández | Matriz de costos, análisis de thresholds, generación de predicciones finales y revisión de resultados para la submission | 25% |

---

## Archivos generados

```text
submission_kaggle_f1.csv
submission_threshold_costo_minimo.csv
submission_threshold_operativo.csv
```

Descripción:

- `submission_kaggle_f1.csv`: archivo recomendado para Kaggle/Cadem.
- `submission_threshold_costo_minimo.csv`: predicciones usando el threshold que minimiza costo esperado.
- `submission_threshold_operativo.csv`: predicciones usando el threshold costo-beneficio más equilibrado.

---

## Notas importantes

- `test.csv` no contiene la variable objetivo `Churn`, por lo que no se utiliza para calcular métricas.
- Las métricas reportadas provienen del conjunto de validación interna.
- Para Kaggle/Cadem se prioriza F1-score.
- La matriz de costos se utiliza como análisis adicional costo-beneficio.
- El threshold 0.12 minimiza costo, pero genera demasiados falsos positivos.
- El threshold 0.40 representa una alternativa operativa más equilibrada.
- El threshold 0.55 se utiliza para la submission oficial porque maximiza F1-score en validación.