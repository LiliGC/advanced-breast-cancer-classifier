# 🩺 Clasificador Avanzado de Cáncer de Mama

[![CI/CD to Google Cloud](https://github.com/LiliGC/advanced-breast-cancer-classifier/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/LiliGC/advanced-breast-cancer-classifier/actions/workflows/ci-cd.yml)

Una aplicación completa de Machine Learning para la clasificación de tumores de mama usando Random Forest optimizado, con API Flask y frontend interactivo en Streamlit, todo orquestado con Docker.

## ✨ Características Principales
- **Dataset**: Wisconsin Breast Cancer Dataset (WDBC) de UCI.
### 🤖 Modelo de ML Mejorado
- **Random Forest optimizado** con GridSearchCV.
- **Validación cruzada** y métricas comprehensivas (Accuracy, ROC-AUC, Recall).
- **Análisis de importancia de features** para explicabilidad.
- **Generación automática de visualizaciones**: Curva ROC, matriz de confusión, etc.
- **Creación de casos de ejemplo** (benigno, maligno, límite) basados en datos reales para demostraciones consistentes.

### 🌐 API REST Robusta
- **Endpoints**: `health`, `model/info`, `features`, `predict` (individual y batch), y `visualizations`.
- **Validación robusta** de datos de entrada y manejo de errores.
- **Cálculo de contribución de features** para cada predicción.
- **Logging** para seguimiento y depuración.

### 🎯 Frontend Interactivo
- **4 pestañas especializadas**:
  - 🔍 Predicción con casos predefinidos
  - 📊 Análisis de features del dataset
  - 🎛️ Modo manual con sliders interactivos
  - 📈 Visualizaciones del modelo
- **Gráficos interactivos** con Plotly
- **Métricas en tiempo real**
- **Interfaz intuitiva y profesional** con CSS personalizado.

### 🐳 Contenerización con Docker
- **Pipeline reproducible** con Docker Compose.
- **4 servicios definidos**: `training`, `api`, `frontend`, y `tester`.
- **Inicio automatizado** con un solo script (`start.sh`).
- **Pruebas de integración** automatizadas contra la API en un entorno aislado.

### 🚀 CI/CD y Despliegue en la Nube
- **Pipeline Automatizado con GitHub Actions**: Cada `push` a la rama `main` dispara un flujo de trabajo que automáticamente entrena, prueba, construye y despliega la aplicación.
- **Despliegue en Google Cloud**: La aplicación se despliega en **Google Cloud Run**, una plataforma serverless que escala automáticamente, garantizando alta disponibilidad y eficiencia.
- **Registro de Imágenes en Artifact Registry**: Las imágenes de Docker se versionan y almacenan de forma segura en Google Artifact Registry.


## 🎥 Demostración

A continuación se presentan breves demostraciones de la aplicación.

### Frontend Interactivo

Un vistazo rápido a la interfaz de usuario, mostrando una predicción en tiempo real y el dashboard de resultados.

**➡️ [Ver Demo Completa del Frontend en YouTube](https://youtu.be/sKUN_uGA1IQ)**

### Backend y Pruebas

Demostración de la interacción entre la API, los contenedores y las pruebas automatizadas.

**➡️ [Ver Demo Completa del Backend en YouTube](https://youtu.be/5-SuQPW3JMw)**

## 🚀 Instalación y Uso Rápido (con Docker)

Este proyecto está diseñado para ejecutarse con Docker, lo que simplifica enormemente la configuración y garantiza la reproducibilidad.

**Requisitos:**
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) instalado y en ejecución.
- Un terminal que soporte scripts de shell (Git Bash o WSL en Windows).

**Pasos para ejecutar:**

1.  **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/LiliGC/advanced-breast-cancer-classifier.git
    cd advanced-breast-cancer-classifier
    ```

2.  **Ejecutar el Script de Orquestación:**
    El método recomendado es utilizar el script `start.sh` que gestiona todo el ciclo de vida de la aplicación local.
    Este script actúa como un orquestador que ejecuta los comandos de `docker-compose` en la secuencia correcta para asegurar un inicio limpio y ordenado.

    ```bash
    # En Windows (usando Git Bash o WSL) o en Linux/macOS
    ./start.sh
    ```
    Este script se encargará de:
    - Limpiar entornos Docker previos.
    - Construir las imágenes necesarias.
    - Ejecutar el contenedor de entrenamiento para generar los artefactos del modelo.
    - Levantar la API y el Frontend.
    - Ejecutar las pruebas de integración contra la API.

3.  **Acceder a la Aplicación:**
    - **Frontend:** Abre tu navegador y ve a `http://localhost:8501`
    - **API Health Check:** `http://localhost:5000/health`

### 🛠️ Flujo de Desarrollo Diario

Una vez que la aplicación ha sido construida y entrenada con `start.sh`, no necesitas ejecutar todo el script cada vez.

- **Para iniciar solo la API y el Frontend (sin re-entrenar):**
  ```bash
  docker-compose up -d api frontend
  ```
- **Para detener todos los servicios:**
  ```bash
  docker-compose down
  ```
#### Limpieza del Entorno

Si deseas eliminar los contenedores, volúmenes y las imágenes construidas para liberar espacio en tu disco:

1.  **Detener y eliminar contenedores y volúmenes:**
    ```bash
    docker-compose down -v
    ```

2.  **Eliminar las imágenes construidas por el proyecto:**
    ```bash
    # Reemplaza 'breast_cancer_project' si tu carpeta de proyecto tiene otro nombre
    docker rmi $(docker images -q 'breast_cancer_project*')
    ```
## 📋 Estructura del Proyecto

```
breast_cancer_project/
📁 .github/
│   ├── 📁 workflows
│   │   ├── ci-cd.yml          # Flujo de trabajo de CI/CD
├── 📁 breast_cancer_app/
│   ├── 📁 model
│   │   ├── train_model.py     # Script de entrenamiento
│   ├── 📁 api
│   │   ├── api.py              # API Flask
|   ├── 📁 frontend
│   │   ├── frontend.py         # Frontend Streamlit
├── 📁 tests
│   │   ├── test_api.py         # Tests automatizados
├── 📁 requirements/
│   ├── common.txt              # Dependencias comunes
│   ├── api.txt                 # Dependencias de la API
│   ├── frontend.txt            # Dependencias del Frontend
│   └── dev.txt                 # Dependencias de desarrollo
├── 📁 artifacts (automático)
│   ├── model.pkl  # Modelo entrenado
│   ├── feature_info.json         # Info de features
│   ├── model_metrics.json        # Métricas
│   └── *.png                    # Visualizaciones
|___📁 docs/
|    └── DEVELOPMENT_GUIDE.md    # Guía de evolución del desarrollo del proyecto
|── Dockerfile                   # Dockerfile para la API
|── Docker-compose.yml           # Orquestador de contenedores
|── .dockerignore                # Archivos a ignorar en Docker
|── .gitignore                   # Archivos a ignorar en Git
|── start.sh                     # Script de inicio automatizado
|── README.md                    # Documentación del proyecto
```

## 🔧 Componentes Detallados

### 1. 🤖 Entrenamiento del Modelo (`breast_cancer_app/model/train_model.py`)

**Características:**
- **Optimización de hiperparámetros** con GridSearchCV.
- **Validación cruzada** para evaluar la generalización del modelo.
- **Métricas completas**: Accuracy, ROC-AUC, y reporte de clasificación.
- **Análisis de importancia de features** para explicabilidad.
- **Generación automática de visualizaciones**: Curva ROC, matriz de confusión, etc.

**Archivos generados en `artifacts/`:**
- `model.pkl` - El modelo entrenado y serializado.
- `roc_curve.png` - Curva ROC del modelo.
- `confusion_matrix.png` - Matriz de confusión en el set de prueba.

### 2. 🌐 API REST (`breast_cancer_app/api/api.py`)
- **Endpoints**: `health`, `model/info`, `features`, `predict` (individual y batch), y `visualizations`.
- **Validación robusta** de datos de entrada.
- **Cálculo de contribución de features** para cada predicción.
- **Logging** para seguimiento y depuración.

### 3. 🎯 Frontend Interactivo (`breast_cancer_app/frontend/frontend.py`)
- **Navegación por pestañas**: Inicio, Predicción, Análisis del Dataset y Rendimiento del Modelo.
- **Dashboard de resultados** con medidor de confianza y factores clave.
- **Visualizaciones interactivas** con Plotly para explorar el dataset y el modelo.
- **Diseño profesional** y responsivo con CSS personalizado.

## 🛠️ Tecnologías Utilizadas

- **ML**: scikit-learn, pandas, numpy
- **Backend**: Flask, Flask-CORS
- **Frontend**: Streamlit, Plotly, Altair
- **Visualización**: matplotlib, seaborn
- **CI/CD**: GitHub Actions
- **Cloud**: Google Cloud Run, Google Artifact Registry
- **Testing**: requests, pytest
- **Utilidades**: joblib, json

## 🎯 Casos de Uso

1. **🩺 Médicos**: Herramienta de apoyo diagnóstico
2. **🎓 Estudiantes**: Aprender ML aplicado
3. **👨‍💻 Desarrolladores**: Plantilla para proyectos de ML
4. **🔬 Investigadores**: Análisis de features biomédicas

## 📈 Métricas de Rendimiento

El modelo optimizado típicamente alcanza:
- **Accuracy**: >95%
- **ROC-AUC**: >98%
- **Tiempo de predicción**: <100ms
- **Memoria**: <50MB

## 🤝 Contribuir

¡Las contribuciones para mejorar este proyecto son bienvenidas!

**Posibles extensiones:**
- 🧠 **Explicabilidad Avanzada**: Integrar librerías como **SHAP** para generar visualizaciones de contribución más detalladas por predicción.
- 🎨 **Mejoras en la UI/UX**: Añadir temas, mejorar la responsividad o incorporar nuevos tipos de gráficos.
- 📊 **Monitoreo del Modelo**: Integrar la API con **Vertex AI Model Monitoring** para detectar *data drift* y *concept drift* en producción, registrando predicciones en BigQuery.
- 🗄️ **Historial de Predicciones**: Añadir una base de datos (como SQLite o PostgreSQL) para guardar y consultar predicciones pasadas.

## 📞 Troubleshooting y Soporte

Si encuentras algún problema al ejecutar el proyecto localmente, sigue estos pasos:

1.  **Asegúrate de usar el script `start.sh`**: Este script garantiza que el entorno se limpie y se construya en el orden correcto. Es la forma recomendada de iniciar la aplicación.

2.  **Revisar los Logs de Docker**: Si un servicio no se inicia o se comporta de forma extraña, los logs son tu mejor amigo. Abre una nueva terminal y ejecuta:
    ```bash
    # Para ver los logs de la API
    docker-compose logs api

    # Para ver los logs del frontend
    docker-compose logs frontend
    ```

3.  **Reconstrucción Forzada**: Si has hecho cambios en el código y no se reflejan, es posible que necesites forzar una reconstrucción. El script `start.sh` ya lo hace, pero si quieres hacerlo manualmente:
    ```bash
    docker-compose build --no-cache
    ```

4.  **Abrir un Issue**: Si el problema persiste, la mejor manera de reportarlo es creando un "Issue" en el repositorio de GitHub. Por favor, incluye la salida de los logs y los pasos que seguiste.

## 🏗️ CI/CD y Despliegue en la Nube

### Flujo de Trabajo de CI/CD con GitHub Actions

Este proyecto utiliza un pipeline de Integración Continua y Despliegue Continuo (CI/CD) para automatizar todo el ciclo de vida de la aplicación.

1.  **Activación**: El flujo de trabajo se activa con cada `push` a la rama `main`.
2.  **Entrenamiento y Pruebas**: Se ejecuta el script de entrenamiento para generar los artefactos del modelo y se realizan pruebas de integración contra la API.
3.  **Construcción de Imágenes**: Se construyen las imágenes de Docker para la API y el frontend.
4.  **Publicación**: Las imágenes se etiquetan y se suben a Google Artifact Registry.
5.  **Despliegue**: Se despliegan las nuevas versiones de los servicios en Google Cloud Run, actualizando la aplicación sin tiempo de inactividad.

### Despliegue en Google Cloud (Configuración)

El pipeline está preparado para desplegar automáticamente en Google Cloud usando **Workload Identity Federation**, un método seguro que no requiere claves de servicio de larga duración.

**Pasos a seguir en Google Cloud (Resumen):**

1.  **Crear un Proyecto y Habilitar APIs**:
    - Crea un nuevo proyecto en la Consola de Google Cloud.
    - Habilita las siguientes APIs: `Cloud Run API`, `Artifact Registry API`, y `IAM Credentials API`.

2.  **Crear Repositorios en Artifact Registry**:
    - Crea dos repositorios de tipo Docker en Artifact Registry: uno llamado `api-repo` y otro `frontend-repo`.

3.  **Configurar la Identidad para el Pipeline**:
    - **Crea una Cuenta de Servicio (Service Account)**, por ejemplo, `github-actions-deployer`.
    - **Otorga los roles necesarios** a esta cuenta de servicio: `Cloud Run Admin`, `Artifact Registry Writer`, y `Service Account User`.
    - **Crea un Workload Identity Pool y un Provider**. Configura el proveedor para que confíe en tu repositorio de GitHub.
    - **Vincula la Cuenta de Servicio** a la identidad de GitHub, otorgándole el rol `Workload Identity User`. Esto permite que GitHub Actions actúe en nombre de tu cuenta de servicio de forma segura.

**Pasos a seguir en GitHub:**

1.  **Configurar los Secretos**: Ve a `Settings > Secrets and variables > Actions` en tu repositorio y añade los siguientes secretos, que obtuviste de los pasos anteriores en Google Cloud:
    - `GCP_PROJECT_ID`
    - `GCP_WORKLOAD_IDENTITY_PROVIDER`
    - `GCP_SERVICE_ACCOUNT`

2.  **Activar el Job de Despliegue**: Por defecto, el job `deploy` está comentado en `.github/workflows/ci-cd.yml` para evitar ejecuciones fallidas en un repositorio nuevo. Para habilitar el despliegue automático, simplemente descomenta esa sección en el archivo. A partir de ese momento, cada `push` a `main` desplegará la aplicación en la nube.

## 🚨 Consideraciones Importantes

⚠️ **Este es un proyecto educativo/demostrativo**
- No debe usarse para diagnósticos médicos reales
- Siempre consultar con profesionales médicos
- Los resultados son para fines de aprendizaje únicamente

## 🎉 ¡Disfruta explorando el mundo del Machine Learning!

Este proyecto demuestra un pipeline completo de ML desde el entrenamiento hasta el deployment, con todas las mejores prácticas incluidas. ¡Perfecto para aprender y expandir! 🚀