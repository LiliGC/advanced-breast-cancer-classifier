# train_model.py
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os

# Ruta absoluta al directorio de artefactos para que funcione sin importar desde dónde se ejecute
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(MODEL_DIR, '..', '..', 'artifacts'))
if not os.path.exists(ARTIFACTS_DIR):
    os.makedirs(ARTIFACTS_DIR)
    
def load_and_prepare_data():
    """Cargar y preparar el dataset"""
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    
    # Guardar nombres de features y estadísticas para el frontend
    feature_info = {
        'feature_names': data.feature_names.tolist(),
        'feature_descriptions': {
            'mean radius': 'Radio promedio del núcleo',
            'mean texture': 'Textura promedio (desviación estándar de grises)',
            'mean perimeter': 'Perímetro promedio del núcleo',
            'mean area': 'Área promedio del núcleo',
            'mean smoothness': 'Suavidad promedio (variación local en longitudes de radio)',
            'mean compactness': 'Compacidad promedio (perímetro²/área - 1.0)',
            'mean concavity': 'Concavidad promedio (severidad de partes cóncavas)',
            'mean concave points': 'Puntos cóncavos promedio (número de partes cóncavas)',
            'mean symmetry': 'Simetría promedio',
            'mean fractal dimension': 'Dimensión fractal promedio'
        },
        'feature_ranges': {
            name: {'min': float(X[name].min()), 'max': float(X[name].max()), 
                   'mean': float(X[name].mean()), 'std': float(X[name].std())}
            for name in X.columns  # Ahora para TODAS las 30 features
        }
    }
    
    with open(os.path.join(ARTIFACTS_DIR, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    return X, y, data.target_names

def optimize_model(X_train, y_train):
    """Optimizar hiperparámetros del modelo"""
    print("🔍 Optimizando hiperparámetros...")
    
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=5, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"✅ Mejores parámetros: {grid_search.best_params_}")
    print(f"✅ Mejor score (ROC-AUC): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluar el modelo comprehensivamente"""
    print("\n📊 Evaluando modelo...")
    
    # Predicciones
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_pred_proba_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
    
    print(f"Train Accuracy: {train_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Classification report
    print("\nClassification Report:")
    class_report_dict = classification_report(y_test, y_pred_test, output_dict=True)
    print(classification_report(y_test, y_pred_test))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features más importantes:")
    print(feature_importance.head(10))
    
    # Guardar métricas
    metrics = {
        'train_accuracy': float(train_score),
        'test_accuracy': float(test_score),
        'roc_auc': float(roc_auc),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'classification_report': class_report_dict,
        'feature_importance': feature_importance.to_dict('records')
    }
    
    with open(os.path.join(ARTIFACTS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics

def create_visualizations(model, X, X_test, y_test):
    """Crear visualizaciones del modelo"""
    print("\n📈 Creando visualizaciones...")
    
    # Feature importance plot
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=feature_importance.head(15), y='feature', x='importance')
    plt.title('Top 15 Features más Importantes')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC Curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()    
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion Matrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Maligno', 'Benigno'],
                yticklabels=['Maligno', 'Benigno'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Real')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Matriz de Correlación del dataset completo
    plt.figure(figsize=(20, 16))
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.1f', cmap='coolwarm', annot_kws={"size": 8})
    plt.title('Matriz de Correlación de Todas las Características', fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_example_cases(model, X: pd.DataFrame, y: pd.Series):
    """Encontrar y guardar casos de ejemplo robustos para el frontend."""
    print("\n💾 Guardando casos de ejemplo...")
    
    # Predecir probabilidades para todo el dataset
    y_proba = model.predict_proba(X)
    
    # Filtrar por casos correctamente clasificados
    correct_predictions_mask = model.predict(X) == y
    
    # --- Caso Benigno Típico ---
    # De los casos benignos (y=1) correctamente clasificados, encontrar el que tiene la máxima probabilidad de ser benigno (y_proba[:, 1])
    benign_mask = (y == 1) & correct_predictions_mask
    benign_index = y_proba[benign_mask, 1].argmax()
    benign_case = X[benign_mask].iloc[benign_index].tolist()

    # --- Caso Maligno Típico ---
    # De los casos malignos (y=0) correctamente clasificados, encontrar el que tiene la máxima probabilidad de ser maligno (y_proba[:, 0])
    malignant_mask = (y == 0) & correct_predictions_mask
    malignant_index = y_proba[malignant_mask, 0].argmax()
    malignant_case = X[malignant_mask].iloc[malignant_index].tolist()

    # --- Caso Límite ---
    # Encontrar el caso (de cualquier tipo) cuya probabilidad esté más cerca de 0.5
    borderline_index = np.abs(y_proba[:, 1] - 0.5).argmin()
    borderline_case = X.iloc[borderline_index].tolist()

    example_cases = {
        "benign_typical": {
            "features": benign_case,
            "description": "Caso real del dataset clasificado como benigno con alta confianza."
        },
        "malignant_typical": {
            "features": malignant_case,
            "description": "Caso real del dataset clasificado como maligno con alta confianza."
        },
        "borderline_case": {
            "features": borderline_case,
            "description": "Caso real del dataset donde el modelo mostró la mayor incertidumbre."
        }
    }
    
    with open(os.path.join(ARTIFACTS_DIR, 'example_cases.json'), 'w') as f:
        json.dump(example_cases, f, indent=2)

def main():
    print("🚀 Iniciando entrenamiento mejorado del modelo...")
    
    # Cargar datos
    X, y, target_names = load_and_prepare_data()
    print(f"📊 Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} features")
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Entrenar modelo base para comparación
    print("\n🤖 Entrenando modelo base...")
    base_model = RandomForestClassifier(n_estimators=100, random_state=42)
    base_model.fit(X_train, y_train)
    base_score = base_model.score(X_test, y_test)
    print(f"Accuracy modelo base: {base_score:.4f}")
    
    # Optimizar modelo
    optimized_model = optimize_model(X_train, y_train)
    
    # Evaluar modelo optimizado
    metrics = evaluate_model(optimized_model, X_train, X_test, y_train, y_test)
    
    # Crear visualizaciones
    create_visualizations(optimized_model, X, X_test, y_test)
    
    # Guardar casos de ejemplo
    save_example_cases(optimized_model, X, y)
    
    # Guardar modelo final
    model_data = {
        'model': optimized_model,
        'feature_names': X.columns.tolist(),
        'target_names': target_names.tolist(),
        'scaler': None  # No usamos scaler en este caso
    }
    
    joblib.dump(model_data, os.path.join(ARTIFACTS_DIR, "model.pkl"))
    
    print(f"\n✅ Modelo mejorado entrenado y guardado!")
    print(f"📈 Mejora en accuracy: {metrics['test_accuracy'] - base_score:.4f}")
    print("📊 Archivos generados:")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'model.pkl')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'feature_info.json')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'model_metrics.json')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'example_cases.json')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'feature_importance.png')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'roc_curve.png')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png')}")
    print(f"  - {os.path.join(ARTIFACTS_DIR, 'correlation_matrix.png')}")

if __name__ == "__main__":
    main()