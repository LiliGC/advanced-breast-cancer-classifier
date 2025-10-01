# frontend.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime
import os
import base64

# Configuración de la página
st.set_page_config(
    page_title="Clasificador IA de Cáncer de Mama",
    page_icon="🎀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la apariencia
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #b3d9ff;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #ffeaa7;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuración y Conexión a la API ---
BASE_URL = os.environ.get("API_URL", "http://127.0.0.1:5000")

@st.cache_data(ttl=300)
def get_api_health():
    """Verificar el estado de la API"""
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else {}
    except:
        return False, {}

@st.cache_data(ttl=600)
def get_model_info():
    """Obtener información del modelo"""
    try:
        response = requests.get(f"{BASE_URL}/model/info", timeout=10)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

@st.cache_data(ttl=600)
def get_features_info():
    """Obtener información de features"""
    try:
        response = requests.get(f"{BASE_URL}/features", timeout=10)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

@st.cache_data(ttl=600)
def get_json_artifact(filename: str):
    """Obtener un artefacto JSON (como example_cases) desde la API."""
    try:
        # Usamos el endpoint de visualizaciones que también puede servir archivos JSON
        response = requests.get(f"{BASE_URL}/visualizations/{filename}", timeout=10)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

def make_prediction(features):
    """Hacer predicción a través de la API"""
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"features": features},
            timeout=15
        )
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def show_dataset_info():
    """Mostrar información detallada del dataset"""
    st.markdown("""
    <div class="info-box">
    <h3>📊 Dataset: Wisconsin Breast Cancer (Diagnostic)</h3>
    <p><strong>Fuente:</strong> UCI Machine Learning Repository</p>
    <p><strong>Creado por:</strong> Dr. William H. Wolberg, University of Wisconsin Hospitals</p>
    <p><strong>Año:</strong> 1995</p>
    <p><strong>Casos totales:</strong> 569 muestras</p>
    <p><strong>Distribución:</strong> 357 benignos (62.7%) | 212 malignos (37.3%)</p>
    <p><strong>Origen de los Datos:</strong> Las características se computan a partir de imágenes digitalizadas de una <strong>Aspiración con Aguja Fina (AAF)</strong>, un procedimiento de biopsia.</p>
    </div>
    """, unsafe_allow_html=True)

def setup_sidebar(api_info, model_info):
    """Configura la barra lateral mejorada"""
    with st.sidebar:
        st.markdown("### 🤖 Estado del Sistema")
        
        if api_info:
            st.success("✅ API Conectada")
            st.markdown(f"**Modelo:** {api_info.get('model', 'N/A')}")
            st.markdown(f"**Features:** {api_info.get('features_count', 'N/A')}")
            st.markdown(f"**Versión:** {api_info.get('version', 'N/A')}")
        else:
            st.error("❌ API Desconectada")
        
        if model_info and 'metrics' in model_info:
            st.markdown("### 🎯 Métricas del Modelo")
            metrics = model_info['metrics']
            
            # Extraer recall de la clase 'Maligno' (clase '0')
            recall_maligno = metrics.get('classification_report', {}).get('0', {}).get('recall', 0)
            
            st.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.1%}")
            st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
            
            # Mostrar Recall de forma destacada
            st.metric(
                "Recall (Maligno)", 
                f"{recall_maligno:.1%}",
                help="Capacidad del modelo para encontrar TODOS los casos malignos. Un valor alto es crucial."
            )
        
        st.markdown("---")
        st.markdown("""
        ### 👨‍💻 Información del Desarrollador
        **Desarrollado por:** LiliGC  
        **Tecnologías:** 
        - 🤖 scikit-learn
        - 🌐 Flask API
        - 📊 Streamlit
        - 📈 Plotly
        
        ### 🔗 Recursos
        - [Dataset UCI](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
        - [Información Médica](https://www.cancer.org/cancer/breast-cancer.html)
        """)

def create_gauge_chart(value, title, color_scheme="green"):
    """Crear gráfico de gauge mejorado"""
    if color_scheme == "green":
        color = "#00cc96"
        steps_colors = ["#ff6b6b", "#feca57", "#48dbfb", "#00cc96"]
    else:
        color = "#ff6b6b"
        steps_colors = ["#00cc96", "#48dbfb", "#feca57", "#ff6b6b"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 24}},
        delta=None, # Eliminamos el delta para mayor claridad
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color, 'thickness': 0.3},
            'steps': [ # Simplificado para evitar errores de versión
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ]}))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    return fig

def show_home():
    """Página de inicio mejorada"""
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>🩺 Clasificador IA de Cáncer de Mama</h1>
        <p>Sistema inteligente de apoyo diagnóstico usando Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Información del dataset
    show_dataset_info()
    
    # Características principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🤖 Tecnología IA</h3>
        <p>Random Forest optimizado con validación cruzada y análisis de importancia de características</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>⚡ Tiempo Real</h3>
        <p>Predicciones instantáneas con análisis de confianza y factores de decisión</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Visualización</h3>
        <p>Gráficos interactivos y dashboards informativos para análisis completo</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Cómo usar la aplicación
    st.markdown("### 🚀 Cómo usar esta aplicación")
    
    steps_col1, steps_col2 = st.columns([1, 1])
    
    with steps_col1:
        st.markdown("""
        **1. 🔮 Realizar Predicción**
        - Selecciona casos predefinidos o ajusta valores manualmente
        - Obtén diagnóstico instantáneo con nivel de confianza
        - Analiza las características más importantes
        
        **2. 📊 Explorar Dataset**
        - Revisa estadísticas del conjunto de datos
        - Comprende las distribuciones de características
        - Visualiza patrones y tendencias
        """)
    
    with steps_col2:
        st.markdown("""
        **3. 📈 Evaluar Modelo**
        - Examina métricas de rendimiento
        - Visualiza curvas ROC y matrices de confusión
        - Entiende la importancia de cada característica
        
        **4. 🎛️ Experimentar**
        - Modifica parámetros en tiempo real
        - Observa cómo cambian las predicciones
        - Aprende sobre el comportamiento del modelo
        """)

    # Nueva sección para la información médica
    st.markdown("### 🩺 Información Médica sobre Cáncer de Mama")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
        <h4>¿Qué es?</h4>
        <p>Es una enfermedad en la que las células de la mama se multiplican sin control. Es el segundo cáncer más común en mujeres a nivel mundial.</p>
        
        <h4>Síntomas Clave</h4>
        <ul>
            <li>Bulto en la mama o axila</li>
            <li>Cambios en tamaño, forma o piel</li>
            <li>Secreción del pezón</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-box">
        <h4>Factores de Riesgo</h4>
        <ul>
            <li>Edad (>50 años) y antecedentes familiares</li>
            <li>Mutaciones genéticas (BRCA1, BRCA2)</li>
            <li>Estilo de vida (obesidad, alcohol)</li>
        </ul>
        
        <h4>Detección Temprana</h4>
        <p>Es crucial. Incluye autoexámenes, exámenes clínicos y mamografías regulares (especialmente a partir de los 40 años).</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disclaimer médico
    st.markdown("""
    <div class="warning-box">
    <h3>⚠️ Descargo de Responsabilidad Médica</h3>
    <p><strong>IMPORTANTE:</strong> Esta aplicación es una herramienta educativa y de demostración tecnológica. 
    <strong>NO debe utilizarse para diagnósticos médicos reales.</strong></p>
    <ul>
    <li>Los resultados son generados por un modelo de IA con fines educativos</li>
    <li>NO reemplaza la consulta con profesionales médicos cualificados</li>
    <li>Para cualquier preocupación médica, consulte siempre a un médico</li>
    <li>La detección temprana requiere exámenes médicos profesionales</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🤔 ¿Por qué no puedo subir una mamografía?")
    st.markdown("""
    <div class="info-box">
    <p>Este modelo de IA fue entrenado con <strong>datos numéricos</strong> extraídos de un procedimiento de biopsia llamado <strong>Aspiración con Aguja Fina (AAF)</strong>. No analiza imágenes directamente.</p>
    <ul>
        <li><strong>Análisis de Imágenes:</strong> Requeriría un tipo de modelo completamente diferente (una Red Neuronal Convolucional o CNN) entrenado con miles de imágenes de mamografías.</li>
        <li><strong>Análisis de Datos Tabulares:</strong> Este modelo se especializa en encontrar patrones en las mediciones de las células (radio, textura, etc.) que ya han sido extraídas por un patólogo.</li>
    </ul>
    <p>Ambos enfoques son válidos en la IA médica, pero abordan el problema desde ángulos diferentes.</p>
    </div>
    """, unsafe_allow_html=True)

def show_prediction_tab():
    """Tab de predicción mejorado"""
    st.header("🔮 Predicción Inteligente")
    
    # ¡CORRECCIÓN! Usar session_state para recordar el método de entrada
    if 'prediction_method' not in st.session_state:
        st.session_state.prediction_method = "🎯 Casos Predefinidos"

    # Selector de método
    st.session_state.prediction_method = st.radio(
        "**Método de entrada de datos:**",
        ["🎯 Casos Predefinidos", "🎛️ Ajuste Manual Interactivo"],
        horizontal=True,
        key="prediction_method_selector",
        index=0 if st.session_state.prediction_method == "🎯 Casos Predefinidos" else 1
    )
    
    # Obtener información de features
    features_info = get_features_info()
    
    if st.session_state.prediction_method == "🎯 Casos Predefinidos":
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            case_type = st.selectbox(
                "**Selecciona un caso clínico de ejemplo:**",
                ["🟢 Benigno Típico", "🔴 Maligno Típico", "🟡 Caso Límite"],
                help="Estos son casos reales del dataset seleccionados por su claridad o ambigüedad."
            )
        # Generar features según el caso
        example_cases = get_json_artifact("example_cases.json")
        if example_cases:
            if "Benigno" in case_type:
                case_data = example_cases.get("benign_typical", {})
            elif "Maligno" in case_type:
                case_data = example_cases.get("malignant_typical", {})
            elif "Límite" in case_type:
                case_data = example_cases.get("borderline_case", {})
            
            features = case_data.get("features", [])
            case_description = case_data.get("description", "No se pudo cargar el caso.")
            st.info(f"📝 **Descripción del caso:** {case_description}")
        else:
            st.warning("⚠️ No se pudo cargar información de features. Usando valores por defecto.")
            features = np.random.normal(15, 3, 30).tolist()
    
    else:  # Ajuste Manual
        st.markdown("---")
        st.markdown("### 🎛️ Control Manual de Características")
        st.info("💡 **Instrucciones:** Mueve los deslizadores para simular un caso clínico. Luego, presiona el botón de predicción para ver el análisis del modelo.")
        
        if features_info and 'feature_ranges' in features_info:
            manual_features = {}

            # Crear dos columnas para los sliders
            col1, col2 = st.columns(2)
            
            # ¡CORRECCIÓN CLAVE! Iterar en el orden correcto
            ordered_feature_names = features_info.get('feature_names', list(features_info['feature_ranges'].keys()))
            for i, feature_name in enumerate(ordered_feature_names):
                ranges = features_info['feature_ranges'][feature_name]
                container = col1 if i % 2 == 0 else col2
                
                with container:
                    description = features_info.get('feature_descriptions', {}).get(feature_name, 'Característica médica del tumor')
                    
                    # Formatear el nombre para mejor legibilidad
                    display_name = feature_name.replace('mean ', '').replace('_', ' ').title()
                    
                    manual_features[feature_name] = st.slider(
                        f"**{display_name}**",
                        min_value=float(ranges['min']),
                        max_value=float(ranges['max']),
                        value=float(ranges['mean']),
                        step=(ranges['max'] - ranges['min']) / 200,
                        key=f"manual_{feature_name}",
                        help=f"{description}\nRango: {ranges['min']:.2f} - {ranges['max']:.2f}"
                    )
            
            # ¡CORRECCIÓN CLAVE! Asegurar el orden correcto de las features.
            # Recolectar los valores en el mismo orden que 'ordered_feature_names'.
            features = [manual_features[name] for name in ordered_feature_names]
            
            # Botón para resetear valores
            if st.button("🔄 Resetear a Valores Promedio"):
                st.rerun()
        else:
            st.error("❌ No se pudo cargar la información de features.")
            return
    
    # Predicción
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 **REALIZAR PREDICCIÓN**", type="primary", use_container_width=True): # st.button aún no soporta 'width'
            with st.spinner("🧠 Analizando características del tumor..."):
                time.sleep(1)  # Pequeña pausa para UX
                success, result = make_prediction(features)
                
                if success and 'prediction' in result:
                    st.session_state['last_prediction'] = result
                    st.session_state['prediction_time'] = datetime.now()
                else:
                    st.error(f"❌ Error en la predicción: {result.get('error', 'Error desconocido')}")
    
    # Mostrar resultados
    if 'last_prediction' in st.session_state:
        st.markdown("---")
        st.markdown("## 📋 Dashboard de Resultados")
        
        result = st.session_state['last_prediction']
        prediction_time = st.session_state.get('prediction_time', datetime.now())
        
        # Información principal
        prediction_label = result.get('interpretation', 'Desconocido')
        confidence = result.get('confidence', 0) * 100
        is_benign = result.get('prediction') == 1
        
        # Layout principal de resultados
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        
        with result_col1:
            if is_benign:
                st.markdown("""
                <div class="success-box">
                <h3>✅ BENIGNO</h3>
                <p>Tumor no canceroso</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: #f8d7da; padding: 1rem; border-radius: 10px; border: 1px solid #f5c6cb; margin: 1rem 0;">
                <h3>⚠️ MALIGNO</h3>
                <p>Tumor potencialmente canceroso</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"**Tiempo de análisis:** {prediction_time.strftime('%H:%M:%S')}")
        
        with result_col2:
            # Gauge de confianza
            gauge_color = "green" if is_benign else "red"
            gauge_fig = create_gauge_chart(confidence, "Nivel de Confianza (%)", gauge_color)
            st.plotly_chart(gauge_fig)
        
        with result_col3:
            # Probabilidades
            probs = result.get('prediction_proba', [0, 0])
            prob_maligno = probs[0] * 100
            prob_benigno = probs[1] * 100
            
            st.markdown("**Probabilidades:**")
            st.markdown(f"🔴 Maligno: {prob_maligno:.1f}%")
            st.markdown(f"🟢 Benigno: {prob_benigno:.1f}%")
            
            # Gráfico de barras pequeño
            prob_fig = go.Figure()
            prob_fig.add_trace(go.Bar(
                x=['Maligno', 'Benigno'],
                y=[prob_maligno, prob_benigno],
                marker_color=['#ff6b6b', '#51cf66']
            ))
            prob_fig.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=40),
                showlegend=False,
                yaxis_title="Probabilidad (%)"
            )
            st.plotly_chart(prob_fig)
        
        # Análisis de características importantes
        if 'feature_contributions' in result:
            st.markdown("### 🎯 Características Más Influyentes en la Decisión")
            
            contrib_df = pd.DataFrame(result['feature_contributions'][:8])  # Top 8
            
            fig = px.bar(
                contrib_df,
                x='contribution',
                y='feature',
                orientation='h',
                color='contribution',
                color_continuous_scale='viridis',
                title="Contribución de cada característica a la predicción"
            )
            fig.update_layout(
                height=400,
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Nivel de Contribución",
                yaxis_title="Característica"
            )
            st.plotly_chart(fig)

            # ¡CORRECCIÓN! Mover la tabla detallada dentro del bloque 'if'
            with st.expander("📊 Ver Detalles de Todas las Características"):
                detailed_df = pd.DataFrame(result['feature_contributions']) # Usar todos los datos, no solo el top 8
                detailed_df['value'] = detailed_df['value'].round(3)
                detailed_df['contribution'] = detailed_df['contribution'].round(4)
                detailed_df = detailed_df.rename(columns={
                    'feature': 'Característica',
                    'value': 'Valor Medido',
                    'importance': 'Importancia del Modelo',
                    'contribution': 'Contribución a la Decisión'
                })
                st.dataframe(detailed_df)
def show_dataset_analysis():
    """Tab de análisis del dataset mejorado"""
    st.header("🔬 Análisis del Dataset")
    
    # Información del dataset
    show_dataset_info()
    
    features_info = get_features_info()
    
    if not features_info or 'feature_ranges' not in features_info:
        st.error("❌ No se pudo cargar la información del dataset.")
        return
    
    # Estadísticas generales
    st.markdown("### 📈 Estadísticas de las Características")
    
    ranges_df = pd.DataFrame.from_dict(features_info['feature_ranges'], orient='index')
    ranges_df = ranges_df.round(3)
    ranges_df.index.name = 'Característica'
    ranges_df = ranges_df.rename(columns={
        'min': 'Mínimo',
        'max': 'Máximo',
        'mean': 'Media',
        'std': 'Desviación Estándar'
    })
    
    # Añadir rango y coeficiente de variación
    ranges_df['Rango'] = ranges_df['Máximo'] - ranges_df['Mínimo']
    ranges_df['Coef. Variación'] = (ranges_df['Desviación Estándar'] / ranges_df['Media']).round(3)
    
    
    # Visualizaciones interactivas
    st.markdown("### 📊 Visualizaciones Interactivas")
    
    viz_tab1, viz_tab2, viz_tab3 = st.tabs([
        "📈 Distribuciones",
        "🔗 Correlaciones",
        "📋 Comparativas"
    ])
    
    with viz_tab1:
        st.markdown("**Selecciona características para visualizar sus distribuciones:**")
        st.warning("⚠️ **Nota:** Las siguientes visualizaciones se basan en **datos simulados** con la misma media y desviación estándar que el dataset original para fines de demostración.")
        
        feature_names = list(features_info['feature_ranges'].keys())
        selected_features = st.multiselect(
            "Características:",
            options=feature_names,
            default=feature_names[:4],
            key="dist_features",
            help="Selecciona hasta 6 características para mejor visualización"
        )
        
        if selected_features:
            # Box plots
            fig = go.Figure()
            colors = px.colors.qualitative.Set3
            
            for i, feature in enumerate(selected_features):
                ranges = features_info['feature_ranges'][feature]
                
                # Simular distribución normal
                simulated_data = pd.DataFrame({
                    feature: np.clip(np.random.normal(ranges['mean'], ranges['std'], 1000), ranges['min'], ranges['max'])
                })
                
                fig.add_trace(go.Box(
                    y=simulated_data[feature],
                    name=feature.replace('mean ', '').replace('_', ' ').title(),
                    marker_color=colors[i % len(colors)],
                    boxpoints='outliers'
                ))
            
            fig.update_layout(
                title="Distribución de Características Seleccionadas",
                yaxis_title="Valor",
                height=500
            )
            st.plotly_chart(fig)
            
            # Histogramas
            if len(selected_features) <= 4:
                fig_hist = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=[f.replace('mean ', '').replace('_', ' ').title() for f in selected_features[:4]]
                )
                
                for i, feature in enumerate(selected_features[:4]):
                    ranges = features_info['feature_ranges'][feature]
                    simulated_data = pd.DataFrame({
                        feature: np.clip(np.random.normal(ranges['mean'], ranges['std'], 1000), ranges['min'], ranges['max'])
                    })
                    
                    row = i // 2 + 1
                    col = i % 2 + 1
                    
                    fig_hist.add_trace(
                        go.Histogram(x=simulated_data[feature], name=feature, nbinsx=30),
                        row=row, col=col
                    )
                
                fig_hist.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig_hist)
    
    with viz_tab2:
        st.markdown("**Matriz de Correlación**")
        try:
            response = requests.get(f"{BASE_URL}/visualizations/correlation_matrix.png", timeout=10)
            if response.status_code == 200:
                st.info("💡 Esta visualización muestra la correlación de Pearson entre todas las características del dataset. Un valor cercano a 1 (azul) indica una fuerte correlación positiva, mientras que un valor cercano a -1 (rojo) indica una fuerte correlación negativa.")
                st.image(response.content, caption="Matriz de Correlación Real del Dataset")
            else:
                st.error(f"❌ No se pudo cargar la matriz de correlación. Código de estado: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Error de conexión al cargar la matriz de correlación: {e}")
    
    with viz_tab3:
        st.markdown("**Comparación de Características**")
        st.warning("⚠️ **Nota:** Esta visualización utiliza **datos simulados** para la demostración.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature1 = st.selectbox("Primera característica:", feature_names, index=0, key="comp_feature1")
        with col2:
            feature2 = st.selectbox("Segunda característica:", feature_names, index=1, key="comp_feature2")
        
        if feature1 != feature2:
            ranges1 = features_info['feature_ranges'][feature1]
            ranges2 = features_info['feature_ranges'][feature2]
            
            # Scatter plot simulado
            np.random.seed(42)  # Para reproducibilidad
            n_points = 500
            
            # Simular datos correlacionados
            x_data = np.random.normal(ranges1['mean'], ranges1['std'], n_points)
            y_data = np.random.normal(ranges2['mean'], ranges2['std'], n_points)
            
            # Añadir algo de correlación
            correlation_factor = 0.3
            y_data = y_data + correlation_factor * (x_data - ranges1['mean'])
            
            # Clip a los rangos válidos
            x_data = np.clip(x_data, ranges1['min'], ranges1['max'])
            y_data = np.clip(y_data, ranges2['min'], ranges2['max'])
            
            simulated_df = pd.DataFrame({feature1: x_data, feature2: y_data})
            
            fig = px.scatter(
                simulated_df, x=feature1, y=feature2,
                title=f"Relación entre {feature1} y {feature2}",
                opacity=0.6,
                trendline="ols", trendline_color_override="red"
            )
            fig.add_hline(y=ranges2['mean'], line_dash="dash", line_color="red", 
                         annotation_text=f"Media de {feature2}")
            fig.add_vline(x=ranges1['mean'], line_dash="dash", line_color="blue", 
                         annotation_text=f"Media de {feature1}")
            
            st.plotly_chart(fig)

def show_model_performance():
    """Tab de rendimiento del modelo mejorado"""
    st.header("📈 Evaluación del Rendimiento del Modelo")
    
    model_info = get_model_info()
    
    # Métricas principales
    if model_info and 'metrics' in model_info:
        metrics = model_info['metrics']
        
        st.markdown("### 🎯 Métricas de Rendimiento")
        
        # Métricas en cards
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric(
                "Accuracy de Prueba",
                f"{metrics.get('test_accuracy', 0):.1%}",
                delta=f"{(metrics.get('test_accuracy', 0) - metrics.get('train_accuracy', 0)):.2%}",
                help="Porcentaje de predicciones correctas en el conjunto de prueba"
            )
        
        with metric_col2:
            st.metric(
                "ROC-AUC Score",
                f"{metrics.get('roc_auc', 0):.3f}",
                delta="Excelente" if metrics.get('roc_auc', 0) > 0.9 else "Bueno",
                help="Área bajo la curva ROC (0.5 = aleatorio, 1.0 = perfecto)"
            )
        
        with metric_col3:
            st.metric(
                "Validación Cruzada",
                f"{metrics.get('cv_mean', 0):.3f} ± {metrics.get('cv_std', 0):.3f}",
                help="Promedio de validación cruzada con 5 pliegues"
            )
        
        # Métrica de Recall destacada
        recall_maligno = metrics.get('classification_report', {}).get('0', {}).get('recall', 0)
        st.markdown(f"""
        <div class="metric-card" style="border-left-color: #ff6b6b;">
            <h4>🩺 Recall (Sensibilidad) para Casos Malignos</h4>
            <h1 style="text-align: center; color: #ff6b6b;">{recall_maligno:.1%}</h1>
            <p style="text-align: center;">De todos los tumores que eran realmente malignos, el modelo identificó correctamente este porcentaje. <strong>Esta es la métrica más importante para evitar falsos negativos.</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizaciones del modelo
    st.markdown("### 📊 Visualizaciones del Modelo")
    st.info("Las siguientes visualizaciones se generan a partir de la evaluación del modelo en el conjunto de datos de prueba.")

    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("#### 📊 Ranking Interactivo de Features")
        
        if model_info and 'feature_importance' in model_info.get('metrics', {}):
            importance_data = model_info['metrics']['feature_importance'][:12]  # Top 12
            importance_df = pd.DataFrame(importance_data)
            
            # Gráfico de barras horizontal
            fig = px.bar(
                importance_df,
                x='importance',
                y='feature',
                orientation='h',
                color='importance',
                color_continuous_scale='viridis',
                title="Características Más Importantes del Modelo"
            )
            fig.update_layout(
                height=500,
                yaxis={'categoryorder': 'total ascending'},
                xaxis_title="Importancia Relativa",
                yaxis_title="Característica",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            st.plotly_chart(fig)
            
            # Interpretación de la importancia
            with st.expander("🧠 Interpretación de la Importancia de Features"):
                st.markdown("""
                **¿Qué significa la "importancia de features"?**
                            
                La importancia de una característica indica cuánto se apoya el modelo en ella para tomar sus decisiones. Para un `Random Forest`, se calcula midiendo qué tan efectiva es esa característica para **separar correctamente los casos benignos de los malignos** en el árbol de decisión. Un valor alto significa que la característica es crucial para el diagnóstico.
                            
                **Características Clave y su Contexto Médico:**
                
                Generalmente, el modelo aprende que valores más altos en ciertas mediciones son fuertes indicadores de malignidad. Esto se debe a que los tumores cancerosos tienden a ser más grandes, de forma más irregular y a invadir el tejido circundante.
                
                - **`worst radius`, `worst perimeter`, `worst area`**: Miden el **tamaño** del tumor en su estado más agresivo. Un tamaño mayor es un claro indicador de malignidad.
                - **`worst concave points`, `mean concave points`**: Describen la **irregularidad del contorno** celular. Los núcleos de las células cancerosas suelen tener formas más cóncavas e irregulares, un sello distintivo del cáncer.
                - **`worst texture`**: Mide la **variación en la escala de grises** de la textura celular. Los tumores malignos suelen tener una textura más heterogénea y menos uniforme.
                
                En resumen, el modelo ha aprendido a identificar los patrones de crecimiento descontrolado e irregular que caracterizan a un tumor maligno.
                """)
        else:
            st.info("📊 No hay datos de importancia disponibles")

    with viz_col2:
        st.markdown("#### 📈 Curva ROC y Matriz de Confusión")

        # Usar st.tabs para organizar las dos imágenes
        roc_tab, cm_tab = st.tabs(["Curva ROC", "Matriz de Confusión"])

        with roc_tab:
            try:
                response = requests.get(f"{BASE_URL}/visualizations/roc_curve.png", timeout=10)
                if response.status_code == 200:
                    st.image(response.content, caption="Curva ROC (Receiver Operating Characteristic)")
                    st.markdown("""
                    **¿Cómo leer este gráfico?**
                    - **Eje Y (Sensibilidad):** La capacidad del modelo para encontrar casos malignos (¡lo que queremos!).
                    - **Eje X (1 - Especificidad):** La tasa de falsas alarmas (casos benignos incorrectamente marcados como malignos).
                    
                    **Objetivo:** Queremos una curva que suba rápido hacia la **esquina superior izquierda**. Esto significa que el modelo es muy bueno encontrando tumores malignos (`True Positive Rate` alto) sin dar muchas falsas alarmas (`False Positive Rate` bajo).
                    """)
                else:
                    st.error(f"No se pudo cargar la Curva ROC. Código de estado: {response.status_code}")
            except Exception as e:
                st.error(f"Error de conexión al cargar la Curva ROC: {e}")

        with cm_tab:
            try:
                response = requests.get(f"{BASE_URL}/visualizations/confusion_matrix.png", timeout=10)
                if response.status_code == 200:
                    st.image(response.content, caption="Matriz de Confusión")
                    st.markdown("""
                    **¿Cómo leer esta tabla?**
                    - **Diagonal Principal (Arriba-Izquierda a Abajo-Derecha):** Estos son los **aciertos**.
                        - **Arriba-Izquierda (Verdaderos Malignos):** Casos malignos que el modelo predijo correctamente como malignos.
                        - **Abajo-Derecha (Verdaderos Benignos):** Casos benignos que el modelo predijo correctamente como benignos.
                    - **Fuera de la Diagonal:** Estos son los **errores**.
                        - **Arriba-Derecha (Falsos Benignos):** ¡El error más peligroso! Casos malignos que el modelo predijo incorrectamente como benignos.
                        - **Abajo-Izquierda (Falsos Malignos):** Casos benignos que el modelo predijo incorrectamente como malignos (falsas alarmas).
                    """)
                else:
                    st.error(f"No se pudo cargar la Matriz de Confusión. Código de estado: {response.status_code}")
            except Exception as e:
                st.error(f"Error de conexión al cargar la Matriz de Confusión: {e}")

        # Explicación de la importancia de features que estaba en la otra columna
        if not (model_info and 'feature_importance' in model_info.get('metrics', {})):
            st.info("📊 No hay datos de importancia disponibles")
    
    # Información técnica del modelo
    st.markdown("### ⚙️ Información Técnica del Modelo")
    
    tech_col1, tech_col2 = st.columns(2)
    
    with tech_col1:
        st.markdown("""
        **🤖 Algoritmo Utilizado:**
        - Random Forest Classifier
        - Optimizado con GridSearchCV
        - Validación cruzada de 5 pliegues
        
        **📊 Características del Dataset:**
        - 569 muestras totales
        - 30 características por muestra
        - 357 casos benignos (62.7%)
        - 212 casos malignos (37.3%)
        """)
    
    with tech_col2:
        if model_info and 'model_params' in model_info:
            params = model_info['model_params']
            st.markdown("**🔧 Parámetros del Modelo:**")
            
            key_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
            for param in key_params:
                if param in params:
                    st.markdown(f"- **{param}**: {params[param]}")
        else:
            st.markdown("""
            **🔧 Optimizaciones Aplicadas:**
            - Búsqueda de hiperparámetros
            - Balanceado de clases
            - Validación robusta
            - Análisis de sobreajuste
            """)
    
    # Interpretación médica
    with st.expander("🩺 Interpretación Médica de los Resultados"):
        st.markdown("""
        ### Contexto Médico de las Predicciones
        
        **Métricas en contexto médico:**
        - **Alta precisión (>95%)**: El modelo identifica correctamente la mayoría de casos
        - **ROC-AUC elevado**: Excelente capacidad de discriminación entre tumores benignos y malignos
        - **Baja tasa de falsos negativos**: Minimiza casos malignos clasificados como benignos
        
        **Limitaciones importantes:**
        - Este modelo es una herramienta de apoyo, no un diagnóstico definitivo
        - Los resultados deben interpretarse junto con evaluación clínica
        - La detección temprana requiere múltiples métodos diagnósticos
        
        **Próximos pasos en un contexto real:**
        1. Correlacionar con síntomas clínicos
        2. Realizar estudios de imagen adicionales
        3. Considerar biopsia si está indicada
        4. Evaluación por especialista en oncología
        """)

def main():
    """Función principal de la aplicación"""
    # Verificar conexión con API
    api_status, api_info = get_api_health()
    
    if not api_status:
        st.error("❌ **Error de Conexión con la API**")
        st.markdown("""
        No se puede establecer conexión con el servidor de la API.
        
        **Pasos para solucionar:**
        1. Asegúrate de que la API esté ejecutándose: `python api_improved.py`
        2. Verifica que esté en el puerto correcto: http://127.0.0.1:5000
        3. Si usas Docker: `docker-compose up`
        
        **Estado actual:** API no disponible en `{}`
        """.format(BASE_URL))
        st.stop()
    
    # Cargar información del modelo
    model_info = get_model_info()
    
    # Sidebar
    setup_sidebar(api_info, model_info)
    
    # ¡CORRECCIÓN! Usar session_state para mantener la pestaña activa
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = "Inicio"

    # ¡CORRECCIÓN! Usar st.radio para controlar las pestañas y mantener el estado
    st.session_state.active_tab = st.radio("Navegación Principal", [
        "🏠 Inicio",
        "🔮 Predicción IA",
        "🔬 Análisis Dataset",
        "📈 Rendimiento Modelo"
    ], key="main_nav", horizontal=True, label_visibility="collapsed")

    if st.session_state.active_tab == "🏠 Inicio":
        show_home()
    elif st.session_state.active_tab == "🔮 Predicción IA":
        show_prediction_tab()
    elif st.session_state.active_tab == "🔬 Análisis Dataset":
        show_dataset_analysis()
    elif st.session_state.active_tab == "📈 Rendimiento Modelo":
        show_model_performance()
    
    # Footer
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**🤖 Tecnología:**")
        st.markdown("Random Forest + Flask + Streamlit")
    
    with footer_col2:
        st.markdown("**📊 Dataset:**")
        st.markdown("Wisconsin Breast Cancer (UCI)")
    
    with footer_col3:
        st.markdown("**👨‍💻 Desarrollador:**")
        st.markdown("LiliGC - Sistema IA Médico")
    
    # Información de última actualización
    if api_info and 'timestamp' in api_info:
        try:
            timestamp = api_info['timestamp']
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            st.caption(f"🕐 Última actualización del sistema: {dt.strftime('%d/%m/%Y %H:%M:%S')}")
        except:
            st.caption("🕐 Sistema activo y funcionando")

if __name__ == "__main__":
    main()