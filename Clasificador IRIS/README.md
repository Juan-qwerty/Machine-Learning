# Clasificador con Regresión Lineal - Dataset Iris

Este proyecto implementa un clasificador utilizando **regresión lineal** sobre el dataset de **Iris**.  
Aunque la regresión lineal no es un modelo de clasificación, se adapta mediante la técnica **One vs Rest (OvR)** para predecir entre las tres especies de flores: *setosa*, *versicolor* y *virginica*.

## 🚀 Pasos principales
1. Se cargan los datos desde `sklearn.datasets`.
2. Se dividen en entrenamiento (80%) y prueba (20%).
3. Se entrena un modelo de regresión lineal envuelto en un clasificador OvR.
4. Se realizan predicciones y se ajustan redondeando los valores a enteros.
5. Se evalúa el modelo con:
   - Matriz de confusión
   - Reporte de métricas (precisión, recall, F1-score)
   - Gráficas comparativas de desempeño por clase

## 📊 Resultados
- El modelo alcanza una **precisión moderada**, siendo la clase *setosa* la mejor clasificada.
- Existen confusiones entre *versicolor* y *virginica*, lo cual refleja las limitaciones de la regresión lineal en clasificación.

## 📦 Requerimientos
- Python 3.x
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn

Instalación de dependencias:
```bash
pip install pandas seaborn matplotlib scikit-learn
