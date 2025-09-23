# -*- coding: utf-8 -*-
# 1. LIBRERÍAS Y CONFIGURACIÓN
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Busca y carga el archivo CSV del dataset.
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
file_path = os.path.join(script_dir, "emails_dataset.csv")
if not os.path.exists(file_path):
    print(f"Error: no se encontró '{file_path}'")
    sys.exit(1)

df_raw = pd.read_csv(file_path, encoding="utf-8")


# 2. PREPARACIÓN DE DATOS Y CARACTERÍSTICAS
# Se asegura de que las columnas de texto no tengan nulos.
df_raw['subject'] = df_raw.get('subject', pd.Series([''] * len(df_raw))).fillna('').astype(str)
df_raw['body'] = df_raw.get('body', pd.Series([''] * len(df_raw))).fillna('').astype(str)

# Junta el asunto y el cuerpo en una sola columna.
df_raw['full_text'] = (df_raw['subject'] + ' ' + df_raw['body']).astype(str)
# Versión en minúsculas para facilitar búsquedas.
df_raw['full_text_lower'] = df_raw['full_text'].str.lower()


# DataFrame que contendrá solo las características para el modelo.
df_features = pd.DataFrame(index=df_raw.index)

# Características básicas del texto.
df_features['word_count'] = df_raw['full_text_lower'].str.split().str.len().fillna(0).astype(int)
df_features['char_count'] = df_raw['full_text_lower'].str.len().fillna(0).astype(int)
df_features['avg_word_length'] = (df_features['char_count'] / df_features['word_count'].replace(0, np.nan)).fillna(0) # Evita división por cero

# Características específicas, comunes en SPAM.
df_features['num_exclamations'] = df_raw['full_text'].str.count('!').fillna(0).astype(int)
pattern_upper = r'\b[A-ZÁÉÍÓÚÑÜ]{2,}\b' # Regex para palabras en mayúsculas
df_features['num_uppercase_words'] = df_raw['body'].str.findall(pattern_upper, flags=re.UNICODE).str.len().fillna(0).astype(int)
df_features['num_links'] = df_raw['full_text_lower'].str.count(r'https?://|www\.').fillna(0).astype(int)
df_features['num_numbers'] = df_raw['full_text_lower'].str.count(r'\d').fillna(0).astype(int)

# Proporción de caracteres "raros".
special_char_count = df_raw['full_text_lower'].str.count(r'[^a-zA-Z0-9\s]').fillna(0).astype(int)
df_features['special_char_ratio'] = (special_char_count / df_features['char_count'].replace(0, np.nan)).fillna(0)

# Detecta palabras clave (0 si no está, 1 si está).
df_features['contains_free'] = df_raw['full_text_lower'].str.contains(r'\b(free|gratis|win|premio)\b', regex=True).astype(int)
df_features['contains_money'] = df_raw['full_text_lower'].str.contains(r'(\$|usd|\b(money|dinero|inversi[oó]n|inversion)\b)', regex=True).astype(int)
df_features['contains_click'] = df_raw['full_text_lower'].str.contains(r'\b(click|clic|haz clic|accede|reclama)\b', regex=True).astype(int)

# Función para estandarizar la etiqueta a 0 (HAM) o 1 (SPAM).
def map_label(x):
    if pd.isna(x): return 0
    s = str(x).strip().upper()
    if s in ('SPAM', '1', 'TRUE', 'T', 'YES', 'Y', 'S'): return 1
    try:
        return 1 if float(s) == 1.0 else 0
    except Exception:
        return 1 if 'SPAM' in s else 0

df_features['label'] = df_raw['label'].apply(map_label)


# 3. LIMPIEZA FINAL Y PREPARACIÓN
# Reemplaza infinitos y NaN por 0.
df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
df_features.fillna(0, inplace=True)

# Separa las características (X) de la etiqueta a predecir (y).
X = df_features.drop(columns=['label'])
y = df_features['label'].astype(int)

# Chequeos rápidos para evitar errores.
assert X.isnull().sum().sum() == 0, "Quedan NaNs en X"
assert set(y.unique()).issubset({0, 1}), "Labels distintos de 0/1 detectados"

print("Dataset preparado: muestras =", len(X))
print("Distribución de clases:\n", y.value_counts())

# Función para dividir datos, estratificando solo si es posible.
def safe_train_test_split(X, y, test_size=0.2, random_state=None):
    stratify = y if y.value_counts().min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


# 4. FIGURA 1: VISUALIZACIÓN DE UN ÁRBOL DE DECISIÓN
print("\n--- Generando Figura 1: Árbol de Decisión ---")
# Se entrena un modelo pequeño solo para poder dibujarlo.
X_train_viz, X_test_viz, y_train_viz, y_test_viz = safe_train_test_split(X, y, test_size=0.2, random_state=42)
viz_model = DecisionTreeClassifier(max_depth=4, random_state=42)
viz_model.fit(X_train_viz, y_train_viz)

plt.figure(figsize=(20, 12))
plot_tree(viz_model, feature_names=X.columns.tolist(), class_names=['HAM', 'SPAM'], filled=True, rounded=True, fontsize=10, proportion=True)
plt.title("Figura 1: Estructura del Árbol de Decisión (Profundidad: 4 niveles)", fontsize=16, weight='bold', pad=20)
plt.tight_layout()
plt.savefig('arbol_de_decision.png', dpi=150, bbox_inches='tight')
plt.show()


# 5. EXPERIMENTO: 50 ITERACIONES PARA MEDIR ESTABILIDAD
print("\n--- Ejecutando Experimento de 50 Iteraciones ---")
n_iterations = 50
results = {'accuracy': [], 'f1_score': []}

for i in range(n_iterations):
    try:
        # En cada iteración, se usan distintos datos para entrenar y probar.
        X_tr, X_te, y_tr, y_te = safe_train_test_split(X, y, test_size=0.2, random_state=i)
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        
        # Guarda los resultados de esta vuelta.
        results['accuracy'].append(accuracy_score(y_te, y_pred))
        results['f1_score'].append(f1_score(y_te, y_pred))
    except ValueError:
        continue # Si algo falla, salta a la siguiente iteración.

results_df = pd.DataFrame(results)
# El Z-score mide qué tan atípico fue el resultado de cada iteración.
results_df['zscore_f1'] = (results_df['f1_score'] - results_df['f1_score'].mean()) / results_df['f1_score'].std()

print("Estadísticas de rendimiento (50 iteraciones):")
print(results_df.describe())


# 6. GRÁFICOS DE LOS RESULTADOS DEL EXPERIMENTO

# --- FIGURA 2: Evolución del F1-Score ---
print("\n--- Generando Figura 2: Evolución del F1-Score ---")
plt.figure(figsize=(14, 7))
iterations = range(1, len(results_df) + 1)
mean_f1, std_f1 = results_df['f1_score'].mean(), results_df['f1_score'].std()

plt.plot(iterations, results_df['f1_score'], 'o-', linewidth=2.5, markersize=5, color='#2E86AB', label='F1-Score por iteración', alpha=0.8)
plt.axhline(mean_f1, linestyle='--', linewidth=3, color='#A23B72', label=f'Media: {mean_f1:.4f}')
plt.fill_between(iterations, mean_f1-std_f1, mean_f1+std_f1, alpha=0.3, color='#F18F01', label=f'±1 Desviación Estándar ({std_f1:.4f})')

plt.xlabel('Número de Iteración', fontsize=12, weight='bold')
plt.ylabel('F1-Score', fontsize=12, weight='bold')
plt.title('Figura 2: Evolución del F1-Score en 50 Iteraciones del Experimento', fontsize=14, weight='bold', pad=15)
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(max(0.85, results_df['f1_score'].min() - 0.02), min(1.02, results_df['f1_score'].max() + 0.02))
plt.tight_layout()
plt.savefig('evolucion_f1_score.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURA 3: Evolución del Accuracy ---
print("\n--- Generando Figura 3: Evolución del Accuracy ---")
plt.figure(figsize=(14, 7))
mean_acc, std_acc = results_df['accuracy'].mean(), results_df['accuracy'].std()

plt.plot(iterations, results_df['accuracy'], 's-', linewidth=2.5, markersize=5, color='#28AFB0', label='Accuracy por iteración', alpha=0.8)
plt.axhline(mean_acc, linestyle='--', color='#C73E1D', linewidth=3, label=f'Media: {mean_acc:.4f}')
plt.fill_between(iterations, mean_acc-std_acc, mean_acc+std_acc, color='#F26419', alpha=0.2, label=f'±1 Desviación Estándar ({std_acc:.4f})')

plt.xlabel('Número de Iteración', fontsize=12, weight='bold')
plt.ylabel('Accuracy (Exactitud)', fontsize=12, weight='bold')
plt.title('Figura 3: Evolución del Accuracy en 50 Iteraciones del Experimento', fontsize=14, weight='bold', pad=15)
plt.legend(fontsize=10, loc='lower right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.ylim(max(0.85, results_df['accuracy'].min() - 0.02), min(1.02, results_df['accuracy'].max() + 0.02))
plt.tight_layout()
plt.savefig('evolucion_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURA 4: Z-Score del F1-Score ---
print("\n--- Generando Figura 4: Z-Score del F1-Score ---")
plt.figure(figsize=(14, 7))
plt.plot(iterations, results_df['zscore_f1'], 'D-', linewidth=2.5, markersize=4, color='#6A4C93', label='Z-Score del F1', alpha=0.8)
plt.axhline(0, linestyle='--', color='#2EC4B6', linewidth=3, label='Media (Z = 0)')
plt.axhline(1, linestyle=':', color='#FF9F1C', linewidth=2, alpha=0.7, label='Z = +1')
plt.axhline(-1, linestyle=':', color='#FF9F1C', linewidth=2, alpha=0.7, label='Z = -1')
plt.axhline(2, linestyle=':', color='#E71D36', linewidth=1.5, alpha=0.5, label='Z = ±2')
plt.axhline(-2, linestyle=':', color='#E71D36', linewidth=1.5, alpha=0.5)

plt.xlabel('Número de Iteración', fontsize=12, weight='bold')
plt.ylabel('Z-Score', fontsize=12, weight='bold')
plt.title('Figura 4: Análisis de Z-Score del F1-Score en 50 Iteraciones', fontsize=14, weight='bold', pad=15)
plt.legend(fontsize=10, loc='upper right')
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('zscore_analisis.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURA 5: Distribución de métricas ---
print("\n--- Generando Figura 5: Distribución de Métricas ---")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1) # Gráfico de F1-Score
sns.histplot(results_df['f1_score'], bins=15, kde=True, color='#2E86AB', alpha=0.7, edgecolor='black', linewidth=0.5)
plt.axvline(mean_f1, color='#A23B72', linestyle='--', linewidth=2.5, label=f'Media: {mean_f1:.4f}')
plt.axvline(mean_f1 - std_f1, color='#F18F01', linestyle=':', linewidth=2, label=f'±1σ: {mean_f1-std_f1:.4f}')
plt.axvline(mean_f1 + std_f1, color='#F18F01', linestyle=':', linewidth=2)
plt.xlabel('F1-Score', fontsize=11, weight='bold')
plt.ylabel('Frecuencia', fontsize=11, weight='bold')
plt.title('Distribución del F1-Score', fontsize=12, weight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2) # Gráfico de Accuracy
sns.histplot(results_df['accuracy'], bins=15, kde=True, color='#28AFB0', alpha=0.7, edgecolor='black', linewidth=0.5)
plt.axvline(mean_acc, color='#C73E1D', linestyle='--', linewidth=2.5, label=f'Media: {mean_acc:.4f}')
plt.axvline(mean_acc - std_acc, color='#F26419', linestyle=':', linewidth=2, label=f'±1σ: {mean_acc-std_acc:.4f}')
plt.axvline(mean_acc + std_acc, color='#F26419', linestyle=':', linewidth=2)
plt.xlabel('Accuracy', fontsize=11, weight='bold')
plt.ylabel('Frecuencia', fontsize=11, weight='bold')
plt.title('Distribución del Accuracy', fontsize=12, weight='bold')
plt.legend(fontsize=9)
plt.grid(True, alpha=0.3)

plt.suptitle('Figura 5: Distribución de Métricas de Rendimiento en 50 Iteraciones', fontsize=14, weight='bold', y=1.02)
plt.tight_layout()
plt.savefig('distribucion_metricas.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURA 6: Matriz de confusión ---
print("\n--- Generando Figura 6: Matriz de Confusión ---")
# Se usa el modelo de la mejor iteración para la matriz.
best_idx = int(results_df['f1_score'].idxmax())
X_tr, X_te, y_tr, y_te = safe_train_test_split(X, y, test_size=0.2, random_state=best_idx)
best_model = DecisionTreeClassifier(max_depth=5, random_state=42)
best_model.fit(X_tr, y_tr)
y_pred_best = best_model.predict(X_te)

fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_te, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['HAM', 'SPAM'])
disp.plot(cmap=plt.cm.Blues, values_format=None, ax=ax, colorbar=False)

# Añade conteos y porcentajes a la matriz.
total = cm.sum()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        count = cm[i, j]
        pct = count / total * 100 if total > 0 else 0.0
        text = f"{count}\n({pct:.1f}%)"
        text_color = "white" if count > (cm.max() / 2) else "black"
        ax.text(j, i, text, ha='center', va='center', fontsize=12, fontweight='bold', color=text_color)

ax.set_xlabel('Predicted label', fontsize=12)
ax.set_ylabel('True label', fontsize=12)
ax.set_title('Figura 6: Matriz de Confusión - Mejor Iteración\n(Con porcentajes)', fontsize=14, weight='bold', pad=12)
plt.tight_layout()
plt.savefig('matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.show()

# --- FIGURA 7: Importancia de características ---
print("\n--- Generando Figura 7: Importancia de Características ---")
importancias = best_model.feature_importances_
indices = np.argsort(importancias)[::-1] # Ordena de más a menos importante

caracteristicas_ordenadas = [X.columns[i] for i in indices if importancias[i] > 0.001]
importancias_ordenadas = [importancias[i] for i in indices if importancias[i] > 0.001]

# Nombres más claros para el gráfico.
nombres_amigables = {
    'contains_free': 'Palabras promocionales\n(free, gratis, win, premio)', 'num_links': 'Cantidad de enlaces\nURL en el correo',
    'contains_click': 'Llamadas a acción\n(click, clic, accede, reclama)', 'contains_money': 'Términos financieros\n(money, dinero, $, inversión)',
    'num_exclamations': 'Cantidad de signos\nde exclamación (!)', 'special_char_ratio': 'Proporción de\ncaracteres especiales',
    'num_uppercase_words': 'Palabras escritas\nen mayúsculas', 'num_numbers': 'Cantidad de\nnúmeros presentes',
    'word_count': 'Número total\nde palabras', 'char_count': 'Número total\nde caracteres',
    'avg_word_length': 'Longitud promedio\nde las palabras'
}
nombres_plot = [nombres_amigables.get(col, col) for col in caracteristicas_ordenadas]

plt.figure(figsize=(12, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importancias_ordenadas)))
bars = plt.barh(range(len(importancias_ordenadas)), importancias_ordenadas, color=colors, edgecolor='black', alpha=0.8)

plt.yticks(range(len(importancias_ordenadas)), nombres_plot, fontsize=10)
plt.xlabel('Importancia Relativa', fontsize=12, weight='bold')
plt.title('Figura 7: Importancia de Características para la Detección de SPAM\n(Árbol de Decisión - CART)', fontsize=14, weight='bold', pad=20)

for i, (bar, importancia) in enumerate(zip(bars, importancias_ordenadas)):
    plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, f'{importancia:.3f}', ha='left', va='center', fontsize=9, weight='bold')

plt.gca().invert_yaxis() # La más importante arriba
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.xlim(0, max(importancias_ordenadas) * 1.15 if importancias_ordenadas else 1)
plt.tight_layout()
plt.savefig('importancia_caracteristicas.png', dpi=150, bbox_inches='tight')
plt.show()


# 7. RESUMEN FINAL DEL EXPERIMENTO
print("\n" + "="*70)
print("RESUMEN FINAL DEL EXPERIMENTO - CLASIFICADOR DE SPAM")
print("="*70)

best_f1 = results_df['f1_score'].max()
worst_f1 = results_df['f1_score'].min()
coef_variacion = (std_f1 / mean_f1) * 100 if mean_f1 != 0 else 0
z_scores = results_df['zscore_f1']
outside_2std = np.sum(np.abs(z_scores) > 2)

print(f"• Exactitud (Accuracy) promedio: {mean_acc:.4f} ± {std_acc:.4f}")
print(f"• F1-Score promedio: {mean_f1:.4f} ± {std_f1:.4f}")
print(f"• Mejor F1-Score obtenido: {best_f1:.4f}")
print(f"• Peor F1-Score obtenido: {worst_f1:.4f}")
print(f"• Rango de F1-Score: [{worst_f1:.4f}, {best_f1:.4f}]")

print(f"\n• Coeficiente de variación (F1): {coef_variacion:.2f}%")
print(f"• Iteración con mejor rendimiento: #{best_idx + 1}")
print(f"• Iteraciones fuera de ±2σ: {outside_2std}/{len(results_df)}")
print(f"• Porcentaje dentro de ±2σ: {(len(results_df)-outside_2std)/len(results_df)*100:.1f}%")

print(f"\nTop 5 características más importantes:")
for i, idx in enumerate(indices[:5]):
    if importancias[idx] > 0:
        nombre_amigable = nombres_amigables.get(X.columns[idx], X.columns[idx]).replace('\n', ' ')
        print(f"  {i+1}. {nombre_amigable}: {importancias[idx]:.3f}")

print("\nProceso completado. Todas las gráficas han sido guardadas como PNG.")