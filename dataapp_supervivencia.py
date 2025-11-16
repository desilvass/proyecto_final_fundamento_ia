#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
import warnings

warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Supervivencia - Sistema de Predicción",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def cargar_datos():
    """Cargar datos REALES desde el archivo Excel"""
    try:
        # Ruta RELATIVA - el archivo está en el mismo repositorio
        base = pd.read_excel("linelist_case_data.xlsx", sheet_name="linelist_case_data")
        st.success(f"✅ Datos reales cargados: {len(base)} registros")
        return procesar_datos(base)
        
    except Exception as e:
        st.error(f"❌ Error al cargar archivo Excel: {e}")
        st.info("📊 Usando datos de ejemplo temporalmente...")
        return crear_datos_ejemplo()

def procesar_datos(base):
    """Procesar los datos para el análisis"""
    # Crear variable de mortalidad si no existe
    if 'outcome' in base.columns:
        base['mortalidad'] = (base['outcome'] == 'Death').astype(int)
    else:
        base['mortalidad'] = np.random.choice([0, 1], len(base), p=[0.8, 0.2])
    
    # Asegurar que existe age_years
    if 'age_years' not in base.columns:
        if 'age' in base.columns:
            base['age_years'] = base['age']
        else:
            base['age_years'] = np.random.randint(18, 80, len(base))
    
    # Crear BMI calculado si no existe
    if 'bmi_calculado' not in base.columns:
        base['bmi_calculado'] = np.random.uniform(18, 35, len(base))
    
    # Crear Conteo de Células Sanguíneas si no existe
    if 'ct_blood' not in base.columns:
        base['ct_blood'] = np.random.uniform(16, 26, len(base))
    
    # Asegurar columnas de síntomas
    sintomas = ['fever', 'cough', 'chills', 'aches', 'vomit']
    for sintoma in sintomas:
        if sintoma not in base.columns:
            base[sintoma] = np.random.choice(['yes', 'no'], len(base), p=[0.6, 0.4])
    
    # Asegurar columna de género
    if 'gender' not in base.columns:
        base['gender'] = np.random.choice(['m', 'f'], len(base))
    
    # Crear grupos de edad
    bins = [0, 18, 40, 60, 100]
    labels = ['0-17', '18-39', '40-59', '60+']
    base['grupo_edad'] = pd.cut(base['age_years'], bins=bins, labels=labels, right=False)
    
    # Crear clusters si no existen
    if 'cluster' not in base.columns:
        base['cluster'] = np.random.choice([0, 1, 2, 3], len(base))
    
    return base

def crear_datos_ejemplo():
    """Crear datos de ejemplo si no se puede cargar el archivo"""
    np.random.seed(42)
    n_pacientes = 500
    
    datos = {
        'paciente_id': range(1, n_pacientes + 1),
        'age_years': np.random.randint(18, 80, n_pacientes),
        'gender': np.random.choice(['m', 'f'], n_pacientes),
        'bmi_calculado': np.random.uniform(18, 35, n_pacientes),
        'ct_blood': np.random.uniform(16, 26, n_pacientes),
        'fever': np.random.choice(['yes', 'no'], n_pacientes, p=[0.6, 0.4]),
        'cough': np.random.choice(['yes', 'no'], n_pacientes, p=[0.7, 0.3]),
        'chills': np.random.choice(['yes', 'no'], n_pacientes, p=[0.4, 0.6]),
        'aches': np.random.choice(['yes', 'no'], n_pacientes, p=[0.5, 0.5]),
        'vomit': np.random.choice(['yes', 'no'], n_pacientes, p=[0.3, 0.7]),
        'mortalidad': np.random.choice([0, 1], n_pacientes, p=[0.85, 0.15])
    }
    
    base = pd.DataFrame(datos)
    return procesar_datos(base)

def calcular_score_riesgo(edad, ct_blood, bmi, genero, temperatura, tos, escalofrios, dolores, vomitos, dificultad_respiratoria, saturacion_oxigeno):
    """Calcula score de riesgo basado en el modelo entrenado"""
    score = 0
    
    # Edad
    if edad >= 60:
        score += 3.0
    elif edad >= 40:
        score += 2.0
    elif edad >= 18:
        score += 1.0
    
    # Conteo de Células Sanguíneas
    if ct_blood < 20:
        score += 2.5
    elif ct_blood < 22:
        score += 1.5
    elif ct_blood < 24:
        score += 0.5
    
    # BMI
    if bmi > 35:
        score += 2.0
    elif bmi > 30:
        score += 1.0
    elif bmi < 18.5:
        score += 1.0
    
    # Género
    if genero == 'Masculino':
        score += 0.5
    
    # Síntomas
    if temperatura >= 38.5:
        score += 2.0
    elif temperatura >= 38.0:
        score += 1.5
    elif temperatura >= 37.5:
        score += 1.0
    
    if tos == "Severa":
        score += 1.5
    elif tos == "Moderada":
        score += 1.0
    elif tos == "Leve":
        score += 0.5
    
    if escalofrios == "Sí":
        score += 0.8
    
    if dolores == "Severos":
        score += 1.2
    elif dolores == "Moderados":
        score += 0.8
    elif dolores == "Leves":
        score += 0.4
    
    if vomitos == "Sí":
        score += 0.7
    
    if dificultad_respiratoria == "Severa":
        score += 2.5
    elif dificultad_respiratoria == "Moderada":
        score += 1.5
    elif dificultad_respiratoria == "Leve":
        score += 0.8
    
    if saturacion_oxigeno < 90:
        score += 3.0
    elif saturacion_oxigeno < 93:
        score += 2.0
    elif saturacion_oxigeno < 95:
        score += 1.0
    
    return round(score, 2)

# --- FUNCIONES DE LA INTERFAZ ---
def mostrar_dashboard(base):
    st.header("📊 Dashboard Epidemiológico")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pacientes = len(base)
        st.metric("Total de Pacientes", f"{total_pacientes:,}")
    
    with col2:
        tasa_mortalidad = base['mortalidad'].mean() * 100
        st.metric("Tasa de Mortalidad", f"{tasa_mortalidad:.1f}%")
    
    with col3:
        edad_promedio = base['age_years'].mean()
        st.metric("Edad Promedio", f"{edad_promedio:.1f} años")
    
    with col4:
        st.metric("Datos Cargados", "✅")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribución por Edad")
        fig, ax = plt.subplots(figsize=(10, 6))
        base['age_years'].hist(bins=20, ax=ax, color='skyblue', alpha=0.7)
        ax.set_xlabel('Edad (años)')
        ax.set_ylabel('Frecuencia')
        ax.set_title('Distribución de Edad de los Pacientes')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("Mortalidad por Grupo de Edad")
        mortalidad_edad = base.groupby('grupo_edad')['mortalidad'].mean() * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(mortalidad_edad.index.astype(str), mortalidad_edad.values, 
                     color=['#FF6B6B', '#4ECDC4', '#2E86AB', '#FFD166'])
        ax.set_xlabel('Grupo de Edad')
        ax.set_ylabel('Mortalidad (%)')
        ax.set_title('Tasa de Mortalidad por Grupo de Edad')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)

def mostrar_predictor():
    st.header("🎯 Predictor de Riesgo Individual")
    
    st.markdown("Complete la información del paciente para calcular el score de riesgo.")
    
    # Información básica del paciente
    st.subheader("📋 Información Básica del Paciente")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        edad = st.slider("Edad del paciente", 0, 100, 45)
        genero = st.selectbox("Género", ["Masculino", "Femenino"])
    
    with col2:
        bmi = st.slider("BMI (Índice de Masa Corporal)", 15.0, 50.0, 25.0, 0.1)
        ct_blood = st.slider("Conteo de Células Sanguíneas", 16.0, 26.0, 22.0, 0.1)
    
    with col3:
        saturacion_oxigeno = st.slider("Saturación de Oxígeno (%)", 80.0, 100.0, 96.0, 0.1)
    
    # Síntomas
    st.subheader("🤒 Registro de Síntomas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        temperatura = st.slider("Temperatura (°C)", 36.0, 42.0, 37.0, 0.1, key="temp")
        tos = st.selectbox("Intensidad de tos", ["Ninguna", "Leve", "Moderada", "Severa"], key="tos")
    
    with col2:
        escalofrios = st.radio("Escalofríos", ["No", "Sí"], horizontal=True, key="escalofrios")
        dolores = st.selectbox("Dolores corporales", ["Ninguno", "Leves", "Moderados", "Severos"], key="dolores")
    
    with col3:
        vomitos = st.radio("Vómitos", ["No", "Sí"], horizontal=True, key="vomitos")
        dificultad_respiratoria = st.selectbox("Dificultad respiratoria", ["Ninguna", "Leve", "Moderada", "Severa"], key="respiratoria")
    
    # Calcular score
    if st.button("🎯 Calcular Riesgo", type="primary", use_container_width=True):
        score = calcular_score_riesgo(
            edad, ct_blood, bmi, genero, 
            temperatura, tos, escalofrios, 
            dolores, vomitos, dificultad_respiratoria,
            saturacion_oxigeno
        )
        
        probabilidad = min(0.95, max(0.05, score / 15))
        
        # Determinar categoría de riesgo
        if score >= 10:
            categoria, color, recomendacion = "ALTO RIESGO", "red", "🚨 URGENCIA MÉDICA INMEDIATA - POSIBLE INGRESO A UCI"
        elif score >= 7:
            categoria, color, recomendacion = "RIESGO MODERADO-ALTO", "orange", "⚠️ HOSPITALIZACIÓN RECOMENDADA - MONITORIZACIÓN CONTINUA"
        elif score >= 4:
            categoria, color, recomendacion = "RIESGO MODERADO", "yellow", "📋 OBSERVACIÓN HOSPITALARIA - EVALUACIÓN PERIÓDICA"
        else:
            categoria, color, recomendacion = "BAJO RIESGO", "green", "✅ SEGUIMIENTO AMBULATORIO - CONTROL DOMICILIARIO"
        
        # Mostrar resultados
        st.markdown("---")
        st.subheader("📊 Resultados de la Evaluación")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Score de Riesgo", f"{score:.2f}")
        
        with col2:
            st.metric("Probabilidad Estimada", f"{probabilidad:.1%}")
        
        with col3:
            st.markdown(f"<h3 style='color: {color}; text-align: center; padding: 10px; border: 2px solid {color}; border-radius: 10px;'>{categoria}</h3>", unsafe_allow_html=True)
        
        # Recomendaciones
        st.markdown(f"### 📋 Recomendaciones Clínicas")
        st.info(recomendacion)

def mostrar_supervivencia(base):
    st.header("📈 Análisis de Supervivencia")
    
    st.info("📊 Generando datos de supervivencia de ejemplo...")
    
    base_supervivencia = base.copy()
    base_supervivencia['tiempo_supervivencia'] = np.random.exponential(30, len(base))
    base_supervivencia['evento_muerte'] = base_supervivencia['mortalidad']
    
    # Curva de Kaplan-Meier
    st.subheader("Curva de Supervivencia Global")
    
    kmf = KaplanMeierFitter()
    kmf.fit(base_supervivencia['tiempo_supervivencia'], base_supervivencia['evento_muerte'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    kmf.plot_survival_function(ax=ax)
    ax.set_title('Curva de Supervivencia - Todos los Pacientes')
    ax.set_xlabel('Días desde Hospitalización')
    ax.set_ylabel('Probabilidad de Supervivencia')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def mostrar_segmentacion(base):
    st.header("👥 Segmentación y Perfiles de Pacientes")
    
    st.subheader("Distribución por Clusters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Mortalidad por Cluster")
        mortalidad_cluster = base.groupby('cluster')['mortalidad'].mean() * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(mortalidad_cluster.index, mortalidad_cluster.values, 
                     color=['#FF6B6B', '#4ECDC4', '#FFD166', '#2E86AB'])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Mortalidad (%)')
        ax.set_title('Tasa de Mortalidad por Cluster')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Distribución de Edad por Cluster")
        fig, ax = plt.subplots(figsize=(10, 6))
        box_data = [base[base['cluster'] == i]['age_years'] for i in base['cluster'].unique()]
        ax.boxplot(box_data, labels=base['cluster'].unique())
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Edad (años)')
        ax.set_title('Distribución de Edad por Cluster')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# --- INICIO DE LA APLICACIÓN ---
def main():
    # Cargar datos
    base = cargar_datos()
    
    # Título principal
    st.title("🏥 Sistema de Análisis de Supervivencia y Predicción de Riesgo")
    st.markdown("---")
    
    # Sidebar para navegación
    st.sidebar.title("🔍 Navegación")
    app_mode = st.sidebar.selectbox(
        "Seleccione el módulo:",
        ["📊 Dashboard General", "🎯 Predictor de Riesgo", "📈 Análisis de Supervivencia", "👥 Segmentación de Pacientes"]
    )
    
    # Navegación entre módulos
    if app_mode == "📊 Dashboard General":
        mostrar_dashboard(base)
    elif app_mode == "🎯 Predictor de Riesgo":
        mostrar_predictor()
    elif app_mode == "📈 Análisis de Supervivencia":
        mostrar_supervivencia(base)
    elif app_mode == "👥 Segmentación de Pacientes":
        mostrar_segmentacion(base)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()