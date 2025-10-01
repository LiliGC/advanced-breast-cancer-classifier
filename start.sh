#!/bin/bash

echo "🚀 Iniciando el pipeline completo de Breast Cancer App..."

# Detener y eliminar contenedores, redes y volúmenes antiguos para un inicio limpio
echo "🧹 Limpiando entorno Docker anterior..."
docker-compose down -v --remove-orphans

# Construir las imágenes de Docker
echo "🏗️ Construyendo imágenes Docker..."
docker-compose build

# Levantar los servicios en modo detached (segundo plano)
# El servicio de 'training' se ejecutará primero y luego se detendrá.
# Los servicios 'api' y 'frontend' se iniciarán después.
echo "🚀 Desplegando servicios con Docker Compose..."
docker-compose up -d --build training

# Esperar a que el entrenamiento termine y luego iniciar el resto
echo "⏳ Esperando a que el entrenamiento del modelo finalice..."
docker wait breast-cancer-training

echo "✅ Entrenamiento completado. Iniciando API, Frontend y Pruebas..."
docker-compose up -d api frontend

# Ejecutar las pruebas de integración contra la API
# Usamos 'run' para ejecutar una tarea única sin detener otros servicios.
echo "🧪 Ejecutando pruebas de integración con 'docker-compose run'..."
docker-compose run --rm tester

echo "✅ Pruebas finalizadas."

# Mostrar logs del frontend para que el usuario vea el progreso y la URL
echo "🌟 Frontend y API están en ejecución."
echo "Puedes acceder al frontend en: http://localhost:8501"
echo "Puedes acceder a la API en: http://localhost:5000/health (para health check)"
echo "Mostrando logs del frontend (presiona Ctrl+C para salir)..."
docker-compose logs -f frontend