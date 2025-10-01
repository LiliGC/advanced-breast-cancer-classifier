# test_api.py
import requests
import os

# Obtener la URL de la API desde una variable de entorno, con un valor por defecto para local.
BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:5000")

# --- Datos de prueba ---
# Un caso benigno real del dataset
BENIGN_CASE = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 1.852, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]
# Un caso maligno real del dataset
MALIGNANT_CASE = [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
# Un caso con datos incorrectos
INVALID_CASE = [1, 2, 3]


# --- Funciones de Prueba para Pytest ---

def test_health_check():
    """Prueba que el endpoint /health responde correctamente."""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "API en funcionamiento"
    assert data["model_loaded"] is True

def test_model_info():
    """Prueba que el endpoint /model/info devuelve la información esperada."""
    response = requests.get(f"{BASE_URL}/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "feature_names" in data
    assert "metrics" in data
    assert len(data["feature_names"]) == 30

def test_predict_benign():
    """Prueba una predicción para un caso benigno."""
    response = requests.post(f"{BASE_URL}/predict", json={"features": BENIGN_CASE})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 1  # 1 para Benigno
    assert data["interpretation"] == "Benigno (sano)"
    assert "confidence" in data
    assert "feature_contributions" in data

def test_predict_malignant():
    """Prueba una predicción para un caso maligno."""
    response = requests.post(f"{BASE_URL}/predict", json={"features": MALIGNANT_CASE})
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == 0  # 0 para Maligno
    assert data["interpretation"] == "Maligno (canceroso)"
    assert "confidence" in data
    assert len(data["feature_contributions"]) > 0

def test_predict_invalid_input_shape():
    """Prueba que la API maneja correctamente un input con forma incorrecta."""
    response = requests.post(f"{BASE_URL}/predict", json={"features": INVALID_CASE})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Se esperan 30 features" in data["error"]

def test_predict_missing_features_key():
    """Prueba que la API maneja correctamente un JSON sin la clave 'features'."""
    response = requests.post(f"{BASE_URL}/predict", json={"data": BENIGN_CASE})
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Se requiere 'features'" in data["error"]

def test_predict_batch():
    """Prueba el endpoint de predicción en lote."""
    payload = {
        "samples": [
            BENIGN_CASE,
            MALIGNANT_CASE
        ]
    }
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["summary"]["total_samples"] == 2
    assert data["summary"]["benign_count"] == 1
    assert data["summary"]["malignant_count"] == 1