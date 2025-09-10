import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np

# 1. Se carga el dataset
try:
    df = pd.read_csv('spam_ham_dataset (1).csv')
except FileNotFoundError:
    print("El archivo 'spam_ham_dataset (1).csv' no fue encontrado.")
    exit()


# 2. Análisis de Correlacion
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlacion de Caracteristicas')
plt.savefig('correlation_matrix.png')
plt.close()

# 3. Preparar los datos para el modelo
# Se usaron todas las features en el dataset para la detección de patrones en correos spam.
X = df.drop('spam_label', axis=1)
y = df['spam_label']

# se dividen los datos en conjuntos de entrenamiento y prueba 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# 4. Construir y entrenar el modelo de Regresion Logistica
# se eligio Regresion Logistica porque la variable objetivo (spam_label) es binaria 0 o 1, lo cual es un problema de clasificacion y no de regresion lineal.
model = LogisticRegression(max_iter=1000) # Se puede aumentar el max_iter (epocas) para asegurar la convergencia
model.fit(X_train, y_train)


# 5. Realizar predicciones y evaluar
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1] # Probabilidades para la clase 1 (spam)

# calculamos el F1-Score
f1 = f1_score(y_test, y_pred)
print(f"F1-Score del modelo: {f1:.4f}")

# calculamos la matriz de confusion
cm = confusion_matrix(y_test, y_pred)

# Se construye la figura de la matriz de confusion 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham (No Spam)', 'Spam'],
            yticklabels=['Ham (No Spam)', 'Spam'])
plt.xlabel('Prediccion')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusion')
plt.savefig('confusion_matrix.png')
plt.close()


# 6. Analisis de Importancia de Características
# Los coeficientes del modelo de regresion logistica indican la importancia de cada caracteristica.
importances = model.coef_[0]
feature_names = X.columns

# Crear un DataFrame para que se vea mejor
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Calcular la importancia en porcentaje (basado en el valor absoluto de los coeficientes)
total_importance = np.sum(np.abs(importances))
feature_importance_df['Percentage'] = (np.abs(feature_importance_df['Importance']) / total_importance) * 100
feature_importance_df = feature_importance_df.sort_values(by='Percentage', ascending=False)

print("\nImportancia de las características (en %):")
print(feature_importance_df[['Feature', 'Percentage']])

# Visualizar la importancia de las caracteristicas
plt.figure(figsize=(10, 8))
sns.barplot(x='Percentage', y='Feature', data=feature_importance_df)
plt.title('Importancia de las Características para la Detección de Spam')
plt.xlabel('Importancia (%)')
plt.ylabel('Característica')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# 7. Umbral de decisión
# El modelo usa por defecto un umbral de 0.5. Las probabilidades predichas (y_pred_proba)
# nos dicen "qué tan spam es un correo". Un valor cercano a 1 indica alta probabilidad de ser spam.
# se muestran algunos ejemplos.
results_df = pd.DataFrame({
    'Probabilidad_Spam': y_pred_proba,
    'Clase_Predicha': y_pred,
    'Clase_Real': y_test
})
print("\nEjemplos de probabilidades de spam y clasificación:")
print(results_df.head(10))