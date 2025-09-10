import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Cargar los datos
df = pd.read_csv('spam_ham_dataset_3000.csv')

# --- Modelo con las 2 caracteristicas mas relevantes y datos normalizados ---

print("--- Ejecutando el modelo con las 2 características más importantes y datos normalizados ---")

# Selecciono las 2 características con mayor importancia
top_2_features = [
    'num_uppercase_words',
    'num_links'
]

X_top2 = df[top_2_features]
y = df['spam_label']

# Dividir los datos antes de normalizar
X_train, X_test, y_train, y_test = train_test_split(
    X_top2, y, test_size=0.2, random_state=42, stratify=y
)

# Inicializar el escalador y ajustarlo solo con los datos de entrenamiento
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test) # Usar el mismo scaler para los datos de prueba

# Entrenar y evaluar el modelo
model_top2_normalized = LogisticRegression(max_iter=1000)
model_top2_normalized.fit(X_train_normalized, y_train)
y_pred = model_top2_normalized.predict(X_test_normalized)

# Calcular F1-Score y Matriz de Confusión
f1 = f1_score(y_test, y_pred)
print(f"F1-Score (Top 2 Features, Normalizados): {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)

# Visualizar la Matriz de Confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Ham (No Spam)', 'Spam'],
            yticklabels=['Ham (No Spam)', 'Spam'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión (Top 2 Características, Normalizados)')
plt.savefig('confusion_matrix_top2_normalized.png')
plt.close()