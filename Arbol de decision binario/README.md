# 📧 Clasificador de SPAM con Árboles de Decisión

Este proyecto consiste en un script de Python diseñado para construir un **clasificador de correos electrónicos**. Su función principal es aprender a diferenciar de forma automática entre correos legítimos (**HAM**) y correo no deseado (**SPAM**) utilizando un modelo de **Árbol de Decisión**.

Más allá de solo entrenar un modelo, el script realiza un análisis completo para evaluar qué tan bien funciona, si su rendimiento es estable y cuáles son los factores que más influyen en sus decisiones. El enfoque principal está en el proceso de **ingeniería de características o features**, que es la forma de transformar texto simple en información numérica que un algoritmo pueda interpretar.

-----

## 🎯 Objetivos del Proyecto

  * **Implementar un Clasificador Funcional:** Construir un modelo capaz de recibir un correo y etiquetarlo como SPAM o HAM, utilizando el algoritmo de Árbol de Decisión.
  * **Aplicar Ingeniería de Características:** Diseñar y extraer un conjunto de características numéricas a partir del texto de los correos, con la hipótesis de que estas contendrán las "pistas" que diferencian un correo SPAM de uno legítimo.
  * **Analizar el Rendimiento y la Estabilidad:** No es suficiente que el modelo funcione bien una vez. Se busca comprobar que su rendimiento es consistente a través de múltiples pruebas, utilizando métricas estándar como el F1-Score y el Accuracy.
  * **Garantizar la Interpretabilidad:** Uno de los objetivos clave es poder entender y explicar las decisiones del modelo. Se generarán visualizaciones para responder preguntas como: ¿En qué se fija el modelo para clasificar? ¿Qué características son las más importantes?

-----

## 📊 El Dataset: `emails_dataset.csv`

Para entrenar y probar el modelo, se utilizó un archivo llamado `emails_dataset.csv` que contiene 5000 ejemplos de correos, cada uno con su asunto, cuerpo y una etiqueta que indica si es SPAM o HAM.

### Análisis y Consideraciones sobre los Datos

Es fundamental aclarar que este es un **dataset sintético**, es decir, fue generado por una inteligencia artificial (ChatGPT) para este ejercicio. Esto tiene ventajas y desventajas importantes:

  * **Puntos a Favor:**

      * **Ideal para Aprender:** Al ser un dataset "perfecto" (sin errores, datos faltantes o ruido), nos permite concentrarnos 100% en el algoritmo y la creación de características, sin las distracciones y el trabajo extra que implica la limpieza de datos del mundo real.
      * **Patrones Claros:** Los ejemplos son muy representativos. Los correos SPAM contienen palabras y estructuras muy obvias ("premio", "dinero", "clic aquí"), lo que facilita que el modelo aprenda los patrones básicos de forma efectiva.

  * **Limitaciones Importantes:**

      * **Baja Complejidad:** El SPAM real es mucho más sofisticado. Utiliza sinónimos, imágenes, caracteres invisibles y otras técnicas para engañar a los filtros. Este dataset no presenta ese nivel de desafío.
      * **Rendimiento "Optimista":** Es probable que el alto rendimiento obtenido en este proyecto no se replique exactamente con un dataset de correos reales, ya que estos últimos son mucho más "ruidosos" y complejos.


-----

## 🛠️ Metodología Aplicada

El flujo de trabajo del proyecto se puede dividir en las siguientes etapas:

### 1\. Ingeniería de Características (Feature Engineering)

Esta es la fase más creativa y crucial. Un modelo matemático no entiende de ofertas o reuniones, solo de números. por ello el trabajo es traducir las ideas y patrones del texto a un formato numérico. Para ello, se crearon las siguientes características:

  * **Características de Estructura y Longitud:**

      * `word_count` y `char_count`: Miden la longitud total del correo. La hipótesis es que los correos SPAM pueden ser anormalmente cortos (para llamar la atención rápido) o largos (para esconder palabras clave).
      * `avg_word_length`: La longitud promedio de las palabras. A veces, el SPAM usa palabras extrañas o alargadas.

  * **Características de Estilo y Formato (Pistas de SPAM):**

      * `num_exclamations` y `num_uppercase_words`: El SPAM a menudo abusa de las mayúsculas y los signos de exclamación para generar un falso sentido de urgencia o emoción. Contar su frecuencia es una pista muy útil.
      * `num_links`: La mayoría de los correos SPAM tienen un objetivo: que hagas clic en un enlace. Por tanto, la cantidad de URLs es un indicador muy fuerte.
      * `special_char_ratio`: Una alta proporción de caracteres como `$` `!` `%` `*` suele ser una señal de alerta.

  * **Características Semánticas (Contenido del Mensaje):**

      * Se crearon tres indicadores binarios (0 si no aparece, 1 si aparece) que buscan la presencia de palabras clave muy específicas y de alto impacto:
          * `contains_free`: Busca palabras como "free", "gratis", "win", "premio". 
          * `contains_money`: Busca términos como "money", "dinero", "$", "inversión". 
          * `contains_click`: Detecta "llamadas a la acción" como "click", "clic", "accede", "reclama". 

### 2\. Elección del Modelo: ¿Por qué un Árbol de Decisión?

Con las características ya creadas, se eligió el `DecisionTreeClassifier` de la librería Scikit-learn. Esta elección no fue al azar y se basó en tres ventajas clave para este proyecto:

1.  **Interpretabilidad (Es un modelo de "Caja Blanca"):** Esta es su mayor fortaleza. A diferencia de otros modelos más complejos que funcionan como una "caja negra", un árbol de decisión es totalmente transparente. Genera un conjunto de reglas del tipo "si-entonces" que son fáciles de leer y entender para un ser humano. Esto nos permite no solo saber *qué* predice, sino *por qué* lo predice.
2.  **No Requiere Normalización de Datos:** Los árboles de decisión no se ven afectados por la escala de las características. Esto significa que podemos usar el `word_count` (cuyos valores pueden ser cientos) junto a `contains_free` (que solo es 0 o 1) sin necesidad de aplicar transformaciones complejas a los datos.
3.  **Rapidez de Entrenamiento:** Es un algoritmo computacionalmente ligero y rápido, lo que lo hace perfecto para el siguiente paso: un análisis de estabilidad que requiere entrenar el modelo muchas veces.

### 3\. Experimento de Estabilidad (50 Iteraciones)

Un modelo de Machine Learning puede tener un buen resultado por simple casualidad, dependiendo de qué datos se usaron para entrenar y cuáles para probar. Para asegurarnos de que nuestro clasificador es **estable** y **fiable**, se implementó el siguiente experimento:

1.  Se ejecuta un bucle **50 veces**.
2.  En cada iteración, los datos se mezclan y se dividen de una forma aleatoria y diferente (`random_state` cambia en cada ciclo).
3.  Con cada nueva división, se entrena un árbol de decisión desde cero y se evalúa su rendimiento con las métricas `F1-Score` y `Accuracy`.
4.  Los resultados de las 50 iteraciones se guardan para luego analizar sus estadísticas (media, desviación estándar, etc.).

Este proceso nos da una visión mucho más realista y robusta de la calidad del modelo, ya que promedia su rendimiento en 50 escenarios diferentes, mitigando el factor suerte.

-----

## 📚 Justificación de las Librerías Utilizadas

Cada librería externa cumple un rol fundamental y fue elegida por ser un estándar en el campo de la ciencia de datos.

  * `pandas` y `numpy`: Son la columna vertebral para la manipulación de datos en Python. **Pandas** es indispensable para leer el archivo `emails_dataset.csv` y trabajar con los datos en una estructura tabular (DataFrame). **NumPy** proporciona las herramientas para realizar cálculos numéricos de manera eficiente.
  * `scikit-learn`: Es la librería de Machine Learning más popular de Python. De ella se utilizan componentes esenciales:
      * `DecisionTreeClassifier`: El algoritmo que implementa el modelo.
      * `train_test_split`: La función correcta para dividir los datos en conjuntos de entrenamiento y prueba de forma estratificada, lo que asegura que la proporción de SPAM/HAM se mantenga en ambas partes.
      * `metrics`: Incluye todas las funciones necesarias para evaluar el modelo, como `f1_score`, `accuracy_score` y `confusion_matrix`.
  * `matplotlib` y `seaborn`: El dúo por excelencia para la visualización de datos. **Matplotlib** es la librería base que permite crear todo tipo de gráficos, mientras que **Seaborn** se construye sobre ella para ofrecer gráficos estadísticos más complejos y con una mejor estética por defecto.
  * `os` y `re`: Son módulos nativos de Python. `os` se usa para manejar las rutas de los archivos de forma que el script funcione en cualquier sistema operativo. `re` (expresiones regulares) es la herramienta que permite buscar patrones de texto de forma avanzada, siendo clave para detectar las palabras clave en los correos.

-----

## 🚀 Cómo Empezar

### Requisitos Previos

Para ejecutar este script, solo necesitas tener **Python 3** instalado, junto con las librerías mencionadas anteriormente.

Puedes instalar todas las dependencias necesarias con un único comando en tu terminal:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Instrucciones de Uso

1.  Clona o descarga este repositorio en tu máquina local.
2.  Asegúrate de que el archivo `emails_dataset.csv` se encuentre en la misma carpeta que el script `ArbolBinario.py`.
3.  Abre una terminal, navega hasta la carpeta del proyecto y ejecuta el script con el siguiente comando:
    ```bash
    python ArbolBinario.py
    ```
4.  El script comenzará a procesar los datos, entrenar los modelos y generar los gráficos. Al finalizar, imprimirá un resumen en la consola y todos los gráficos se guardarán como archivos `.png` en la misma carpeta.

-----

## 🔬 Análisis de los Resultados

El script genera varios gráficos, cada uno diseñado para responder una pregunta específica sobre el modelo.

  * **Gráfico del Árbol de Decisión (`arbol_de_decision.png`):** Esta es la visualización más importante para entender el modelo. Muestra, de forma jerárquica, las reglas que el árbol ha aprendido. Se puede seguir el camino desde la raíz hasta las hojas para ver qué preguntas hace el modelo sobre las características ("¿contiene 'dinero'?", "¿cuántos enlaces tiene?") para llegar a un veredicto.

  * **Gráficos de Evolución (`evolucion_f1_score.png`, `evolucion_accuracy.png`):** Estos gráficos muestran el F1-Score y el Accuracy obtenidos en cada una de las 50 iteraciones. Son clave para evaluar la estabilidad. Una línea de media alta con una banda sombreada (que representa la desviación estándar) muy estrecha, indica que el modelo es robusto y su rendimiento no varía mucho aunque cambien los datos de entrenamiento.

  * **Análisis de Z-Score (`zscore_analisis.png`):** El Z-Score nos dice qué tan "extraño" o "atípico" fue el resultado de una iteración en comparación con el promedio. Un valor muy alto o muy bajo (generalmente por encima de 2 o por debajo de -2) podría indicar una división de datos anómala. Este gráfico permite confirmar visualmente que la mayoría de las ejecuciones tuvieron un rendimiento dentro de lo esperado.

  * **Distribución de Métricas (`distribucion_metricas.png`):** Estos histogramas complementan los gráficos de evolución. Muestran la frecuencia de cada resultado. Una forma de campana centrada en un valor alto (ej. 0.95) sugiere que el modelo consistentemente alcanza ese nivel de rendimiento.

  * **Matriz de Confusión (`matriz_confusion.png`):** Esta tabla es fundamental para entender los errores del modelo. Nos dice no solo cuántas veces acertó, sino cómo se equivocó:

      * **Verdaderos Positivos/Negativos:** Aciertos correctos.
      * **Falsos Positivos:** El error más grave. Son correos legítimos (HAM) que el modelo clasificó incorrectamente como SPAM.
      * **Falsos Negativos:** Correos SPAM que el modelo no detectó y se colaron en la bandeja de entrada.

  * **Importancia de las Características (`importancia_caracteristicas.png`):** Este gráfico de barras es, quizás, el más revelador. Responde a la pregunta: "De todo lo que le dimos al modelo, ¿qué le pareció más útil?". Ordena las características de mayor a menor importancia, permitiendo validar si las hipótesis iniciales eran correctas. En este caso, confirma que el contenido semántico (palabras sobre dinero, promociones, etc.) es mucho más decisivo que otras métricas.

-----

## 🏁 Conclusión

El proyecto cumple con su objetivo de construir un clasificador de SPAM que no solo es funcional, sino también **analizable e interpretable**.

El análisis de estabilidad a través de 50 iteraciones demuestra que el modelo tiene un rendimiento **consistente**, y no es producto de una única ejecución afortunada. Por otro lado, el análisis de importancia de características confirma que el modelo ha sido capaz de aprender patrones lógicos y relevantes, priorizando el contenido semántico de los correos para tomar sus decisiones.
