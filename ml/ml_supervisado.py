# ml/ml_supervisado.py - Modelos de Aprendizaje Supervisado
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModeloSupervisado:
    """
    Implementación de modelos supervisados para el hospital
    Unidad III: Análisis supervisado
    """
    
    def __init__(self, db=None, data_cleaner=None):
        self.db = db
        self.data_cleaner = data_cleaner
        self.modelos = {}
        self.scalers = {}
        self.label_encoders = {}
        self.df_consultas = None
        self.df_pacientes = None
        
    def cargar_datos_limpios(self):
        """Cargar datos ya limpios desde CSV"""
        try:
            self.df_pacientes = pd.read_csv('data/pacientes_limpios.csv')
            self.df_consultas = pd.read_csv('data/consultas_limpias.csv')
            logger.info("Datos cargados desde CSV")
            return self
        except FileNotFoundError:
            logger.warning("No se encontraron datos limpios, cargando desde BD...")
            if self.data_cleaner:
                self.data_cleaner.cargar_datos().limpiar_todos()
                self.df_pacientes = self.data_cleaner.df_pacientes_limpio
                self.df_consultas = self.data_cleaner.df_consultas_limpio
            return self
    
    # ============ MODELO DE CLASIFICACIÓN ============
    
    def preparar_datos_clasificacion(self):
        """
        Preparar datos para el modelo de clasificación (Riesgo de Reingreso)
        """
        logger.info("Preparando datos para clasificación...")
        
        # Unir datos de pacientes y consultas
        df = pd.merge(
            self.df_pacientes,
            self.df_consultas.groupby('id_paciente').agg({
                'id_consulta': 'count',
                'duracion_atencion': 'mean',
                'imc': 'mean'
            }).reset_index(),
            on='id_paciente',
            how='inner'
        )
        
        # Crear variable objetivo: reingreso (simulado)
        np.random.seed(42)
        df['reingreso'] = (
            (df['edad'] > 60) * 0.3 +
            (df['id_consulta'] > 5) * 0.2 +
            (df['imc'] > 30) * 0.15 +
            np.random.normal(0, 0.1, len(df))
        )
        df['reingreso'] = (df['reingreso'] > 0.5).astype(int)
        
        # Seleccionar características
        features = ['edad', 'id_consulta', 'imc', 'duracion_atencion']
        
        # Codificar variables categóricas
        le = LabelEncoder()
        df['genero_encoded'] = le.fit_transform(df['genero'].fillna('No especificado'))
        features.append('genero_encoded')
        
        # Normalizar
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
        
        self.scalers['clasificacion'] = scaler
        self.label_encoders['clasificacion'] = le
        
        y = df['reingreso'].values
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, df
    
    def entrenar_clasificador(self):
        """
        Entrenar modelo Random Forest para clasificación
        """
        logger.info("Entrenando clasificador Random Forest...")
        
        X_train, X_test, y_train, y_test, df = self.preparar_datos_clasificacion()
        
        # Entrenar modelo
        modelo = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        modelo.fit(X_train, y_train)
        
        # Evaluar
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        reporte = classification_report(y_test, y_pred, output_dict=True)
        
        # Calcular importancia de características
        feature_names = ['Edad', 'Consultas Previas', 'IMC', 'Duración Atención', 'Género']
        importancias = modelo.feature_importances_
        
        resultado = {
            'modelo': modelo,
            'accuracy': accuracy,
            'reporte': reporte,
            'importancias': dict(zip(feature_names, importancias)),
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'df': df
        }
        
        self.modelos['clasificador'] = modelo
        self.resultado_clasificacion = resultado
        
        logger.info(f"Clasificador entrenado. Accuracy: {accuracy:.4f}")
        return resultado
    
    # ============ MODELO DE REGRESIÓN ============
    
    def preparar_datos_regresion(self):
        """
        Preparar datos para el modelo de regresión (Predicción de Estancia)
        """
        logger.info("Preparando datos para regresión...")
        
        # Unir datos de pacientes y hospitalizaciones
        df = pd.merge(
            self.df_pacientes,
            self.df_consultas.groupby('id_paciente').agg({
                'id_consulta': 'count',
                'imc': 'mean'
            }).reset_index(),
            on='id_paciente',
            how='inner'
        )
        
        # Simular días de estancia (para el ejemplo)
        np.random.seed(42)
        df['dias_estancia'] = (
            5 + 
            0.05 * df['edad'] +
            0.3 * df['id_consulta'] +
            0.1 * df['imc'] +
            np.random.normal(0, 1, len(df))
        )
        df['dias_estancia'] = df['dias_estancia'].clip(1, 30)
        
        # Seleccionar características
        features = ['edad', 'id_consulta', 'imc']
        
        # Normalizar
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
        
        self.scalers['regresion'] = scaler
        
        y = df['dias_estancia'].values
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, df
    
    def entrenar_regresor(self):
        """
        Entrenar modelo de Regresión Lineal para predicción de estancia
        """
        logger.info("Entrenando regresor lineal...")
        
        X_train, X_test, y_train, y_test, df = self.preparar_datos_regresion()
        
        # Entrenar modelo
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        
        # Evaluar
        y_pred = modelo.predict(X_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        resultado = {
            'modelo': modelo,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred': y_pred,
            'df': df,
            'coeficientes': dict(zip(['Edad', 'Consultas Previas', 'IMC'], modelo.coef_))
        }
        
        self.modelos['regresor'] = modelo
        self.resultado_regresion = resultado
        
        logger.info(f"Regresor entrenado. R²: {r2:.4f}, RMSE: {rmse:.4f}")
        return resultado
    
    # ============ GUARDAR Y CARGAR MODELOS ============
    
    def guardar_modelos(self, directorio='ml/modelos'):
        """Guardar modelos entrenados"""
        os.makedirs(directorio, exist_ok=True)
        
        for nombre, modelo in self.modelos.items():
            joblib.dump(modelo, f'{directorio}/{nombre}.pkl')
        
        for nombre, scaler in self.scalers.items():
            joblib.dump(scaler, f'{directorio}/scaler_{nombre}.pkl')
        
        logger.info(f"Modelos guardados en {directorio}/")
    
    def cargar_modelos(self, directorio='ml/modelos'):
        """Cargar modelos guardados"""
        for nombre in ['clasificador', 'regresor']:
            try:
                self.modelos[nombre] = joblib.load(f'{directorio}/{nombre}.pkl')
                logger.info(f"Modelo {nombre} cargado")
            except:
                logger.warning(f"No se pudo cargar {nombre}")
        
        for nombre in ['clasificacion', 'regresion']:
            try:
                self.scalers[nombre] = joblib.load(f'{directorio}/scaler_{nombre}.pkl')
                logger.info(f"Scaler {nombre} cargado")
            except:
                logger.warning(f"No se pudo cargar scaler_{nombre}")
    
    # ============ PREDICCIONES ============
    
    def predecir_riesgo(self, edad, consultas_previas, imc, genero):
        """
        Predecir riesgo de reingreso con el modelo entrenado
        """
        if 'clasificador' not in self.modelos:
            logger.warning("Modelo clasificador no entrenado")
            return None
        
        # Codificar género
        genero_encoded = self.label_encoders['clasificacion'].transform([genero])[0]
        
        # Preparar datos
        features = np.array([[edad, consultas_previas, imc, consultas_previas * 2, genero_encoded]])
        features_scaled = self.scalers['clasificacion'].transform(features)
        
        # Predecir
        riesgo = self.modelos['clasificador'].predict_proba(features_scaled)[0][1]
        
        return float(riesgo)
    
    def predecir_estancia(self, edad, consultas_previas, imc):
        """
        Predecir días de estancia con el modelo entrenado
        """
        if 'regresor' not in self.modelos:
            logger.warning("Modelo regresor no entrenado")
            return None
        
        # Preparar datos
        features = np.array([[edad, consultas_previas, imc]])
        features_scaled = self.scalers['regresion'].transform(features)
        
        # Predecir
        estancia = self.modelos['regresor'].predict(features_scaled)[0]
        
        return float(max(1, estancia))