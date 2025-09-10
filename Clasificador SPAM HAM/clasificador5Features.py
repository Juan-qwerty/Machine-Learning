import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# Cargar los datos
df = pd.read_csv('fictitious_spam_ham_5000.csv')

# --- 1. Modelo con las 5 caracteristicas mas importantes ---

print("--- Ejecutando el modelo con las 5 características mas importantes ---")

# se seleccionan las 5 características con mayor importancia del análisis anterior
top_5_features = [
    'num_uppercase_words',
    'num_links',
    'num_exclamations',
    'num_numbers',
    'contains_free'
]

X_top5 = df[top_5_features]
y = df['spam_label']

# Dividir los datos
X_train_top5, X_test_top5, y_train_top5, y_test_top5 = train_test_split(
    X_top5, y, test_size=0.2, random_state=42, stratify=y
)

# Entrenar y evaluar el modelo
model_top5 = LogisticRegression(max_iter=1000)
model_top5.fit(X_train_top5, y_train_top5)
y_pred_top5 = model_top5.predict(X_test_top5)

# Calcular F1-Score y Matriz de confusion
f1_top5 = f1_score(y_test_top5, y_pred_top5)
print(f"F1-Score (Top 5 Features): {f1_top5:.4f}")

cm_top5 = confusion_matrix(y_test_top5, y_pred_top5)

# Visualizar la Matriz de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm_top5, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Ham (No Spam)', 'Spam'],
            yticklabels=['Ham (No Spam)', 'Spam'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusión (Top 5 Características)')
plt.savefig('confusion_matrix_top5.png')
plt.close()


# --- 2. Modelo con datos normalizados ---

print("\n--- Ejecutando el modelo con datos normalizados (todas las características) ---")

# Usar todas las características
X_full = df.drop('spam_label', axis=1)
y_full = df['spam_label']

# Dividir los datos ANTES de normalizar para evitar fuga de datos
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y
)

# Inicializar el escalador y ajustarlo solo con los datos de entrenamiento
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train_full)
X_test_normalized = scaler.transform(X_test_full) # Usar el mismo scaler para transformar los datos de prueba

# Entrenar y evaluar el modelo con datos normalizados
model_normalized = LogisticRegression(max_iter=1000)
model_normalized.fit(X_train_normalized, y_train_full)
y_pred_normalized = model_normalized.predict(X_test_normalized)

# Calcular F1-Score y Matriz de Confusion
f1_normalized = f1_score(y_test_full, y_pred_normalized)
print(f"F1-Score (Datos Normalizados): {f1_normalized:.4f}")

cm_normalized = confusion_matrix(y_test_full, y_pred_normalized)

# Visualizar la Matriz de Confusion
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Ham (No Spam)', 'Spam'],
            yticklabels=['Ham (No Spam)', 'Spam'])
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.title('Matriz de Confusion (Datos Normalizados)')
plt.savefig('confusion_matrix_normalized.png')
plt.close()