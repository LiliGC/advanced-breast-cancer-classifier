# api.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import json
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Habilitar CORS para el frontend

# Ruta absoluta al directorio de artefactos para que funcione sin importar desde dónde se ejecute
API_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.abspath(os.path.join(API_DIR, '..', '..', 'artifacts'))

# Cargar modelo y metadata
try:
    model_data = joblib.load(os.path.join(ARTIFACTS_DIR, "model.pkl"))
    model = model_data['model']
    feature_names = model_data['feature_names']
    target_names = model_data['target_names']
    
    # Cargar información adicional
    with open(os.path.join(ARTIFACTS_DIR, 'feature_info.json'), 'r') as f:
        feature_info = json.load(f)
    
    with open(os.path.join(ARTIFACTS_DIR, 'model_metrics.json'), 'r') as f:
        model_metrics = json.load(f)
    
    logger.info("✅ Modelo y metadata cargados exitosamente")
    
except Exception as e:
    logger.error(f"❌ Error cargando modelo: {e}")
    model = None
    feature_names = []
    target_names = []
    feature_info = {}
    model_metrics = {}

@app.route("/health", methods=["GET"])
def health():
    """Endpoint de salud con información del modelo"""
    return jsonify({
        "status": "API en funcionamiento",
        "model": "BreastCancer-RandomForest",
        "features_count": len(feature_names),
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0"
    })

@app.route("/", methods=["GET"])
def index():
    """Página de bienvenida de la API"""
    return jsonify({"message": "Bienvenido a la API del Clasificador de Cáncer de Mama. Visita /health para el estado."})

@app.route("/model/info", methods=["GET"])
def model_info():
    """Información detallada del modelo"""
    if not model:
        return jsonify({"error": "Modelo no cargado"}), 500
    
    return jsonify({
        "feature_names": feature_names,
        "target_names": target_names.tolist() if hasattr(target_names, 'tolist') else target_names,
        "feature_info": feature_info,
        "metrics": model_metrics,
        "model_params": model.get_params()
    })

@app.route("/features", methods=["GET"])
def get_features():
    """Obtener información de features para el frontend"""
    return jsonify(feature_info)

@app.route("/predict", methods=["POST"])
def predict():
    """Hacer predicción con validaciones mejoradas"""
    if not model:
        return jsonify({"error": "Modelo no disponible"}), 500
    
    try:
        data = request.get_json()
        
        if not data or "features" not in data:
            return jsonify({"error": "Se requiere 'features' en el JSON"}), 400
        
        features = np.array(data["features"])
        
        # Validar dimensiones
        if len(features) != len(feature_names):
            return jsonify({
                "error": f"Se esperan {len(feature_names)} features, se recibieron {len(features)}"
            }), 400
        
        # Convertir a DataFrame con nombres de features para evitar warnings
        features_df = pd.DataFrame([features], columns=feature_names)
        
        # Hacer predicción
        prediction = int(model.predict(features_df)[0])
        prediction_proba = model.predict_proba(features_df)[0].tolist()
        confidence = float(max(prediction_proba))
        
        # Interpretación
        label = target_names[prediction] if prediction < len(target_names) else "Desconocido"
        interpretation = "Benigno (sano)" if prediction == 1 else "Maligno (canceroso)"
        
        # Feature importance para esta predicción
        feature_contributions = []
        if hasattr(model, 'feature_importances_'):
            for name, importance, value in zip(feature_names, model.feature_importances_, features):
                # Un cálculo de contribución más robusto: importancia * |valor - media|
                mean_value = feature_info.get('feature_ranges', {}).get(name, {}).get('mean', 0)
                feature_contributions.append({
                    "feature": name,
                    "value": float(value),
                    "importance": float(importance),
                    "contribution": float(importance * abs(value - mean_value))
                })
            
            # Ordenar por contribución
            feature_contributions.sort(key=lambda x: x['contribution'], reverse=True)
        
        result = {
            "prediction": prediction,
            "prediction_proba": prediction_proba,
            "confidence": confidence,
            "label": label,
            "interpretation": interpretation,
            "feature_contributions": feature_contributions[:10],  # Top 10
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Predicción realizada: {interpretation} (confianza: {confidence:.3f})")
        return jsonify(result)
        
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Error en los datos: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({"error": "Error interno en la predicción"}), 500

@app.route("/predict/batch", methods=["POST"])
def predict_batch():
    """Predicción en lote"""
    if not model:
        return jsonify({"error": "Modelo no disponible"}), 500
    
    try:
        data = request.get_json()
        
        if not data or "samples" not in data:
            return jsonify({"error": "Se requiere 'samples' con lista de features"}), 400
        
        samples = np.array(data["samples"])
        
        if samples.shape[1] != len(feature_names):
            return jsonify({
                "error": f"Cada muestra debe tener {len(feature_names)} features"
            }), 400
        
        samples_df = pd.DataFrame(samples, columns=feature_names)
        predictions = model.predict(samples_df).tolist()
        probabilities = model.predict_proba(samples_df).tolist()
        
        results = []
        for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
            results.append({
                "sample_id": i,
                "prediction": int(pred),
                "prediction_proba": proba,
                "confidence": float(max(proba)),
                "interpretation": "Benigno" if pred == 1 else "Maligno"
            })
        
        return jsonify({
            "results": results,
            "summary": {
                "total_samples": len(results),
                "benign_count": sum(1 for r in results if r["prediction"] == 1),
                "malignant_count": sum(1 for r in results if r["prediction"] == 0)
            }
        })
        
    except Exception as e:
        logger.error(f"Error en predicción batch: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/visualizations/<filename>", methods=["GET"])
def get_visualization(filename):
    """Servir visualizaciones generadas durante el entrenamiento"""
    # Lista de archivos seguros que se pueden servir
    allowed_files = [
        'feature_importance.png', 
        'roc_curve.png', 
        'confusion_matrix.png',
        'example_cases.json',
        'correlation_matrix.png' # Añadimos la nueva matriz de correlación
    ]    
    filepath = os.path.join(ARTIFACTS_DIR, filename)
    
    if filename not in allowed_files:
        return jsonify({"error": "Archivo no autorizado"}), 403
    
    if not os.path.exists(filepath):
        return jsonify({"error": "Archivo no encontrado"}), 404

    # Determinar el tipo de contenido (MIME type) basado en la extensión
    mimetype = 'application/json' if filename.endswith('.json') else 'image/png'
    return send_file(filepath, mimetype=mimetype)

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint no encontrado"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == "__main__":
    print("🚀 Iniciando API del Clasificador de Cáncer de Mama...")
    print("✅ Endpoints disponibles:")
    print("  GET  /                     - Mensaje de bienvenida")
    print("  GET  /health               - Health check")
    print("  GET  /model/info           - Información del modelo")
    print("  GET  /features             - Información de features")
    print("  POST /predict              - Predicción individual")
    print("  POST /predict/batch        - Predicción en lote")
    print("  GET  /visualizations/<img> - Visualizaciones")
    
    app.run(host="0.0.0.0", port=5000)