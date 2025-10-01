#!/bin/sh
# wait-for-api.sh

# Abortar si algún comando falla
set -e

# La variable API_URL es pasada desde docker-compose.yml
# Extraemos el host y el puerto
host=$(echo $API_URL | cut -d/ -f3 | cut -d: -f1)
port=$(echo $API_URL | cut -d/ -f3 | cut -d: -f2)

echo "⏳ Esperando a que la API en http://$host:$port esté disponible..."

# Bucle hasta que la API responda al endpoint /health con un código 2xx
until curl --output /dev/null --silent --head --fail http://$host:$port/health; do
    printf '.'
    sleep 1
done

echo "\n✅ API está lista. Ejecutando pruebas..."