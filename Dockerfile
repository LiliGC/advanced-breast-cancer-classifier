# === Etapa 1: Base ===
# Imagen común con Python y dependencias base.
FROM python:3.9-slim-bullseye AS base
WORKDIR /app
# Instala dependencias del sistema y limpia la caché para mantener la imagen ligera.
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copia e instala las dependencias comunes.
COPY requirements/common.txt .
RUN pip install --no-cache-dir -r common.txt


# === Etapa 2: Builder de Dependencias de Desarrollo ===
# Etapa para instalar TODAS las dependencias (dev, api, frontend).
# Usada por 'training' y 'tester' en docker-compose.
FROM base AS dev_builder
COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/dev.txt


# === Etapa 3: Imagen Final de la API ===
# Imagen de producción para la API. Ligera y específica.
FROM base AS api
# Copia las dependencias instaladas de la etapa 'dev_builder'.
COPY --from=dev_builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# ¡CORRECCIÓN CLAVE! Copia también los ejecutables como 'waitress-serve'.
COPY --from=dev_builder /usr/local/bin /usr/local/bin
# Copia solo el código de la API. La imagen NO debe contener artefactos.
COPY breast_cancer_app/api /app/breast_cancer_app/api
EXPOSE 5000
# Comando para ejecutar la API en producción con Waitress.
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "breast_cancer_app.api.api:app"]

# === Etapa 5: Imagen Final del Frontend ===
# Imagen de producción para el Frontend.
FROM base AS frontend
COPY --from=dev_builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# ¡CORRECCIÓN CLAVE! Copia también los ejecutables como 'streamlit'.
COPY --from=dev_builder /usr/local/bin /usr/local/bin
COPY breast_cancer_app/frontend /app/breast_cancer_app/frontend
EXPOSE 8501
CMD ["streamlit", "run", "breast_cancer_app/frontend/frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]


# === Etapa 6: Imagen de Desarrollo ===
# Imagen que contiene todo el código y todas las dependencias.
# Ideal para los servicios 'training' y 'tester'.
FROM base AS dev
COPY --from=dev_builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# ¡CORRECCIÓN CLAVE! Copia también los ejecutables como 'pytest'.
COPY --from=dev_builder /usr/local/bin /usr/local/bin
COPY . .