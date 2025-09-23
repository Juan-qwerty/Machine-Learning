# 📧 Clasificador de SPAM con Árboles de Decisión

Este proyecto es un script en Python que implementa un clasificador de correos electrónicos para distinguir entre **SPAM** y **HAM** (correo legítimo) utilizando un modelo de Árbol de Decisión.

El script no solo entrena un modelo, sino que también realiza un análisis completo de su rendimiento y estabilidad a través de múltiples iteraciones, generando visualizaciones detalladas para interpretar los resultados.

## ✨ Características

  * **Extracción de Características**: Analiza el texto de los correos para extraer características relevantes como:
      * Conteo de palabras y caracteres.
      * Número de enlaces, mayúsculas y signos de exclamación.
      * Presencia de palabras clave comunes en SPAM (`gratis`, `dinero`, `clic`, etc.).
      * Proporción de caracteres especiales.
  * **Modelo**: Utiliza `DecisionTreeClassifier` de la librería `scikit-learn`.
  * **Análisis de Estabilidad**: Ejecuta 50 iteraciones de entrenamiento y prueba para evaluar qué tan robusto es el modelo ante diferentes particiones de datos.
  * **Visualizaciones**: Genera y guarda automáticamente 7 gráficos clave para el análisis, incluyendo:
    1.  La estructura del Árbol de Decisión.
    2.  La evolución del F1-Score y Accuracy.
    3.  La matriz de confusión del mejor modelo.
    4.  Un ranking de las características más importantes.

## 🚀 Cómo Empezar

### Requisitos

Necesitas tener Python 3 instalado, junto con las siguientes librerías:

  * pandas
  * numpy
  * scikit-learn
  * matplotlib
  * seaborn

Puedes instalarlas todas con un solo comando:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Uso

1.  Asegúrate de tener el archivo del dataset `emails_dataset.csv` en la misma carpeta que el script de Python.
2.  Ejecuta el script desde tu terminal:
    ```bash
    python tu_script.py
    ```
3.  ¡Listo\! El script imprimirá en la consola un resumen de los resultados y guardará todas las gráficas como archivos `.png` en la misma carpeta.

## 📊 Resultados

Al ejecutar el script, obtendrás dos tipos de resultados:

1.  **En la consola**: Un resumen estadístico del rendimiento promedio del modelo (Accuracy, F1-Score) y una lista de las 5 características más influyentes.

2.  **Archivos de imagen**: Se generarán los siguientes archivos `.png`:

      * `arbol_de_decision.png`: Muestra visualmente cómo el modelo toma sus decisiones.
      * `evolucion_f1_score.png`: Gráfico de la estabilidad del F1-Score en 50 iteraciones.
      * `evolucion_accuracy.png`: Gráfico de la estabilidad del Accuracy.
      * `zscore_analisis.png`: Muestra las iteraciones con rendimiento atípico.
      * `distribucion_metricas.png`: Histogramas de las métricas de rendimiento.
      * `matriz_confusion.png`: Detalla los aciertos y errores (Falsos Positivos/Negativos) del mejor modelo.
      * `importancia_caracteristicas.png`: Un gráfico de barras que ordena las características de más a menos importantes.