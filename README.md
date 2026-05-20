# Proyecto 1 — ML (CS3061): Clasificación Telco Churn

Proyecto de Machine Learning para predecir **Customer Churn** en clientes de telecomunicaciones.

El problema se plantea como una **clasificación binaria**, donde:

- `Churn = Yes` indica que el cliente abandona la empresa.
- `Churn = No` indica que el cliente permanece.

Se trabajó con los archivos oficiales del reto:

- `train.csv`: contiene las variables predictoras y la variable objetivo `Churn`.
- `test.csv`: contiene las mismas variables predictoras, pero sin `Churn`; se usa para generar predicciones finales.
- `sample_submission.csv`: plantilla usada para respetar el formato y orden de `customerID` esperado por Kaggle/Cadem.

El dataset presenta desbalance de clases: aproximadamente **73.4% `No Churn`** y **26.6% `Churn`**.

---

## Estructura del repositorio

```text
.
├── figures/
│   ├── 01_churn_balance.png
│   ├── 02_churn_by_contract.png
│   ├── 03_tenure_by_churn.png
│   ├── 04_correlations.png
│   ├── 05_confusion_logreg.png
│   ├── 06_threshold_optim.png
│   └── 07_confusion_optim-020.png
│
├── notebooks/
│   └── Proyecto_ML_Churn.ipynb
│
├── train.csv
├── test.csv
├── sample_submission.csv
├── submission_kaggle_yesno.csv
├── submission_threshold_costo_beneficio.csv
├── main.tex
├── PROYECTO_MACHINE_LEARNING___GRUPO_10.pdf
├── requirements.txt
└── README.md
```

---

## Pipeline del proyecto

El notebook implementa el siguiente flujo:

1. **Carga de datos**
   - Lectura de `train.csv`, `test.csv` y `sample_submission.csv`.
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
   - Submission con threshold costo-beneficio.

---

## Resultados clave

### Métricas con threshold base 0.5

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Regresión Logística | 0.7560 | 0.5261 | 0.8094 | 0.6377 |
| Random Forest | 0.7728 | 0.5544 | 0.7324 | 0.6311 |

Con threshold 0.5, **Random Forest obtiene mayor accuracy**, mientras que **Regresión Logística obtiene mayor recall** para la clase `Churn`.

---

### Mejor configuración para Kaggle/Cadem

La métrica oficial de Kaggle/Cadem es **F1-score**, por lo que se seleccionó el modelo y threshold que maximizan F1 en validación.

| Criterio | Modelo | Threshold | F1 validación | Accuracy validación |
|---|---|---:|---:|---:|
| Kaggle/Cadem | Regresión Logística | 0.55 | 0.6496 | 0.7808 |

Archivo generado y aceptado por Kaggle/Cadem:

```text
submission_kaggle_yesno.csv
```

Resultado obtenido en Kaggle/Cadem:

| Public Score | Private Score |
|---:|---:|
| 0.640 | 0.640 |

---

### Mejor configuración costo-beneficio

Para el análisis costo-beneficio, se utilizó una matriz de costos que penaliza más los falsos negativos que los falsos positivos. Bajo esta matriz, el menor costo esperado se obtuvo con **Random Forest** y threshold **0.22**.

| Modelo | Threshold | F1 validación | Costo esperado |
|---|---:|---:|---:|
| Random Forest | 0.22 | 0.5656 | $37,785 |
| Random Forest baseline | 0.50 | 0.6311 | $51,375 |

La matriz de confusión para `RandomForest` con threshold `0.22` fue:

| Real / Predicción | No Churn | Churn |
|---|---:|---:|
| No Churn | 404 | 424 |
| Churn | 14 | 285 |

Esta configuración detecta **285 de 299 churners reales**, dejando solo **14 falsos negativos** en validación.

---

## Análisis de thresholds

Se analizaron distintos thresholds porque el modelo primero produce probabilidades y luego estas se convierten en clases (`Churn` / `No Churn`) usando un umbral de decisión.

| Criterio | Modelo | Threshold | F1 | Costo |
|---|---|---:|---:|---:|
| Costo-beneficio | Random Forest | 0.22 | 0.5656 | $37,785 |
| Baseline | Random Forest | 0.50 | 0.6311 | $51,375 |
| Kaggle/Cadem F1 | Regresión Logística | 0.55 | 0.6496 | — |

El threshold **0.22** responde al criterio costo-beneficio.  
El threshold **0.55** responde al criterio oficial de Kaggle/Cadem, basado en F1-score.

---

## Matriz de costos

Se utilizó la siguiente matriz de costos para el análisis costo-beneficio:

| Real / Predicción | No Churn | Churn |
|---|---:|---:|
| No Churn | $0 | $45 |
| Churn | $420 | $45 |

El costo total esperado se calcula como:

```text
Costo = TN*0 + FP*45 + FN*420 + TP*45
```

Donde:

- **TN**: cliente no se iba y el modelo predijo `No Churn`.
- **FP**: cliente no se iba, pero el modelo predijo `Churn`; implica una campaña innecesaria.
- **FN**: cliente sí se iba, pero el modelo predijo `No Churn`; implica perder un cliente.
- **TP**: cliente sí se iba y el modelo predijo `Churn`; implica realizar una campaña de retención.

El falso negativo tiene el costo más alto porque representa a un cliente que realmente se iba, pero el modelo no lo detectó. En cambio, predecir `Churn` activa una acción comercial, por lo que tanto FP como TP tienen un costo asociado de campaña.

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
submission_kaggle_yesno.csv
```

Resultado en Kaggle/Cadem:

```text
Public Score:  0.640
Private Score: 0.640
```

> Nota: aunque la descripción de la competencia mencionaba formato binario 0/1, el archivo `submission_kaggle_yesno.csv` en formato `Yes/No` fue aceptado correctamente por la plataforma.

---

## Ejecutar el proyecto

### Opción 1: Jupyter Notebook

Abrir y ejecutar:

```text
notebooks/Proyecto_ML_Churn.ipynb
```

Ejecutar las celdas en orden desde el inicio.

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
| Alejandro Marcelo | Preprocesamiento, limpieza de datos, tratamiento de `TotalCharges`, codificación de variables y escalado numérico | 25% |
| Ferrel Valentino Infante Garcia | Modelado, validación cruzada, `GridSearchCV`, comparación entre Regresión Logística y Random Forest, y revisión de métricas | 25% |
| Leonardo Martinez Aquino | Análisis exploratorio de datos, interpretación de gráficos, análisis del desbalance de clases y apoyo en el desarrollo del Ejercicio A | 25% |
| Joel David Miguel Fernández | Matriz de costos, análisis de thresholds, selección del umbral costo-beneficio y generación/revisión de submissions finales | 25% |

---

## Archivos generados

```text
submission_kaggle_yesno.csv
submission_threshold_costo_beneficio.csv
```

Descripción:

- `submission_kaggle_yesno.csv`: archivo aceptado por Kaggle/Cadem, optimizado por F1-score.
- `submission_threshold_costo_beneficio.csv`: predicciones usando el threshold que minimiza el costo esperado bajo la matriz propuesta.

---

## Notas importantes

- `test.csv` no contiene la variable objetivo `Churn`, por lo que no se utiliza para calcular métricas.
- Las métricas reportadas provienen del conjunto de validación interna.
- Para Kaggle/Cadem se prioriza F1-score.
- La matriz de costos se utiliza como análisis adicional costo-beneficio.
- El threshold 0.22 responde al criterio costo-beneficio.
- El threshold 0.55 responde al criterio oficial de F1-score.
- El archivo final aceptado por Kaggle/Cadem fue `submission_kaggle_yesno.csv`.