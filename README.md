# Proyecto 1 — ML (CS3061): Clasificación Telco Churn

Dataset: Telco Customer Churn (IBM, 7\,043 clientes). Clasificación binaria con desbalance ~26.5%.

## Estructura
```
data/raw/                  CSV original (no commitear)
data/processed/            X_train/test, y_train/test (output de preprocessing)
src/preprocessing.py       Limpieza + FE + encoding + split + escalado
src/eda.py                 4 figuras + tabla de features
src/modeling.py            LogReg + RandomForest, GridSearchCV
src/cost_threshold.py      Matriz de costos + barrido de threshold
src/ejercicio_a.py         Ejercicio A (vivienda)
report/informe.tex         Informe LaTeX (compilar en Overleaf)
report/*.tex               Tablas LaTeX incluidas en el informe
report/*.md                Resúmenes narrativos
figures/                   9 PNGs para el informe
run_all.py                 Orquestador
```

## Ejecutar

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run_all.py
```

## Resultados clave

| Modelo | CV F1 | Test F1 | Test Acc |
|---|---|---|---|
| Regresión Logística (C=0.01) | 0.633 | 0.631 | 0.752 |
| Random Forest (depth=10, n=200) | 0.636 | 0.630 | 0.769 |

**Threshold óptimo** (matriz de costos FN=$400, FP=$20, TP=$50, TN=$0):
- $t^*=0.18$ → costo $33,320 (vs $50,760 con $t=0.5$) — **34% de ahorro**.

## Compilar el informe

```bash
# Local con TeX Live:
cd report && pdflatex informe.tex && pdflatex informe.tex

# O Overleaf: subir la carpeta report/ + figures/ entera.
```

## Seed
`SEED = 42` en `src/preprocessing.py`. Replicabilidad garantizada.
