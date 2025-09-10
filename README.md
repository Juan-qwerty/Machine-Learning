# Modelo de Detección de SPAM con Regresión Logística

Este proyecto implementa un modelo de Machine Learning para clasificar correos electrónicos como **SPAM** (correo no deseado) o **HAM** (correo normal) utilizando Regresión Logística con `scikit-learn`.

## 📜 Descripción

El objetivo principal es construir un clasificador binario preciso,  y analizar que caracteristicas son más predictivas y evaluar el rendimiento del modelo bajo diferentes condiciones experimentales. El proceso incluye análisis de datos, entrenamiento del modelo y pruebas de robustez.

## 💾 Dataset

Se utilizó un dataset inicial (`spam_ham_dataset.csv`) que contiene 1,000 registros y 11 características extraídas del contenido de los correos electrónicos. Cada registro está etiquetado como spam (1) o ham (0).

Adicionalmente, se generó un dataset sintético de 5,000 registros (`fictitious_spam_ham_5000.csv`) para futuras pruebas.

### Características (Features)

1.  `word_count`: Número total de palabras.
2.  `char_count`: Número total de caracteres.
3.  `avg_word_length`: Longitud promedio de las palabras.
4.  `num_exclamations`: Cantidad de signos de exclamación.
5.  `num_uppercase_words`: Cantidad de palabras en mayúsculas.
6.  `num_links`: Cantidad de enlaces.
7.  `num_numbers`: Cantidad de números.
8.  `special_char_ratio`: Proporción de caracteres especiales.
9.  `contains_free`: Presencia de la palabra "free".
10. `contains_money`: Presencia de palabras relacionadas con dinero.
11. `contains_click`: Presencia de palabras que incitan a hacer clic.

## 🚀 Cómo Empezar

### Prerrequisitos

Asegúrate de tener Python 3 y las siguientes librerías instaladas:

```bash
pip install pandas scikit-learn seaborn matplotlib
```

### Ejecución

1.  Clona este repositorio.
2.  Coloca los archivos `.csv` en la misma carpeta que los scripts de Python o notebooks.
3.  Ejecuta el script principal para entrenar el modelo y ver los resultados.

## 📊 Resultados y Conclusiones

El modelo de Regresión Logística demostró ser **muy efectivo**, alcanzando un F1-Score de 1.00 en el conjunto de datos de prueba original, lo que indica una clasificación perfecta.

  * **Características más influyentes:** El número de **palabras en mayúsculas**, la cantidad de **enlaces** y el número de **signos de exclamación** fueron los predictores más potentes.
  * **Robustez del modelo:** Los experimentos mostraron que un modelo entrenado con solo las 5 características más importantes mantenía un rendimiento perfecto.
  * **Límites del modelo:** Al reducir las características a solo 2, el rendimiento disminuyó ligeramente (F1-Score de 0.9831), demostrando la importancia de tener un conjunto de características adecuado para capturar todos los patrones.