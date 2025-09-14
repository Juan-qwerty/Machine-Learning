import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# 1. Cargar y preparar los datos
# ================================
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# División en entrenamiento (80%) y de prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Se construye y entrena
# ================================
linear_model = LinearRegression()
ovr_classifier = OneVsRestClassifier(linear_model)
ovr_classifier.fit(X_train, y_train)


# 3. Predicción y evaluación
# ================================
y_pred = ovr_classifier.predict(X_test)

# Como regresión puede devolver valores no enteros, redondeamos
y_pred = y_pred.round().astype(int)

class_names = iris.target_names

print(f"Precisión (Accuracy) del modelo: {accuracy_score(y_test, y_pred):.2f}\n")

# 4. Se visualizan los resultados
# ================================

# --- Matriz de Confusión ---
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Matriz de Confusión')
plt.ylabel('Clase Verdadera')
plt.xlabel('Clase Predicha')
plt.show()

# --- Reporte de clasificación ---
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Excluimos filas de resumen
report_df = report_df.drop(['accuracy', 'macro avg', 'weighted avg'])

# --- Gráficas de métricas ---
colores = ["#2c7cd8", "#3aaa5f", "#7950c4"] 
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', color=colores)
plt.title('Métricas de Clasificación por Clase')
plt.ylabel('Puntuación')
plt.xlabel('Clase de Flor')
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
