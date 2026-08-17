# ml/ml_no_supervisado.py - Modelos de Aprendizaje No Supervisado (CORREGIDO)
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModeloNoSupervisado:
    """
    Implementación de modelos no supervisados para el hospital
    Unidad IV: Análisis no supervisado
    """
    
    def __init__(self, db=None, data_cleaner=None):
        self.db = db
        self.data_cleaner = data_cleaner
        self.modelos = {}
        self.scalers = {}
        self.df_pacientes = None
        self.df_consultas = None
        self.resultado_kmeans = None
        self.resultado_pca = None  # <-- Inicializar correctamente
        self.optimizacion_clusters = None
        
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
                if hasattr(self.data_cleaner, 'df_pacientes_limpio'):
                    self.df_pacientes = self.data_cleaner.df_pacientes_limpio
                if hasattr(self.data_cleaner, 'df_consultas_limpio'):
                    self.df_consultas = self.data_cleaner.df_consultas_limpio
            return self
    
    # ============ K-MEANS CLUSTERING ============
    
    def preparar_datos_clustering(self):
        """
        Preparar datos para clustering de pacientes
        """
        logger.info("Preparando datos para clustering...")
        
        if self.df_pacientes is None or self.df_consultas is None:
            logger.error("No hay datos cargados")
            return None, None
        
        # Agregar métricas por paciente
        metricas = self.df_consultas.groupby('id_paciente').agg({
            'id_consulta': 'count',
            'duracion_atencion': 'mean',
            'imc': 'mean',
            'temperatura': 'mean',
            'frecuencia_cardiaca': 'mean'
        }).reset_index()
        metricas.columns = ['id_paciente', 'total_consultas', 'duracion_promedio', 'imc_promedio', 'temp_promedio', 'fc_promedio']
        
        # Unir con pacientes
        df = pd.merge(
            self.df_pacientes,
            metricas,
            on='id_paciente',
            how='inner'
        )
        
        if len(df) == 0:
            logger.warning("No hay datos para clustering, creando datos sintéticos")
            return self._crear_datos_sinteticos()
        
        # Características para clustering
        features = ['edad', 'total_consultas', 'duracion_promedio', 'imc_promedio']
        
        # Codificar género
        df['genero_encoded'] = df['genero'].map({'Masculino': 0, 'Femenino': 1, 'Otro': 2}).fillna(2)
        features.append('genero_encoded')
        
        # Verificar y manejar NaN
        for col in features:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())
        
        # Normalizar
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
        
        # Eliminar filas con NaN
        nan_rows = np.isnan(X).any(axis=1)
        if nan_rows.any():
            logger.warning(f"Eliminando {nan_rows.sum()} filas con NaN")
            X = X[~nan_rows]
            df = df[~nan_rows]
        
        self.scalers['clustering'] = scaler
        self.df_clustering = df
        
        return X, df
    
    def _crear_datos_sinteticos(self):
        """Crear datos sintéticos para clustering"""
        np.random.seed(42)
        n_samples = 100
        
        df = pd.DataFrame({
            'id_paciente': list(range(1, n_samples + 1)),
            'edad': np.random.randint(18, 90, n_samples),
            'genero': np.random.choice(['Masculino', 'Femenino'], n_samples),
            'total_consultas': np.random.randint(1, 20, n_samples),
            'duracion_promedio': np.random.randint(15, 60, n_samples),
            'imc_promedio': np.random.uniform(18, 40, n_samples),
            'temp_promedio': np.random.uniform(36, 38.5, n_samples),
            'fc_promedio': np.random.randint(60, 100, n_samples)
        })
        
        features = ['edad', 'total_consultas', 'duracion_promedio', 'imc_promedio']
        df['genero_encoded'] = df['genero'].map({'Masculino': 0, 'Femenino': 1}).fillna(0)
        features.append('genero_encoded')
        
        scaler = StandardScaler()
        X = scaler.fit_transform(df[features])
        
        self.scalers['clustering'] = scaler
        self.df_clustering = df
        
        return X, df
    
    def entrenar_kmeans(self, n_clusters=3):
        """
        Entrenar modelo K-Means para segmentación de pacientes
        """
        logger.info(f"Entrenando K-Means con {n_clusters} clusters...")
        
        X, df = self.preparar_datos_clustering()
        
        if X is None or len(X) == 0:
            logger.error("No hay datos para entrenar K-Means")
            return None
        
        # Entrenar modelo
        kmeans = KMeans(
            n_clusters=min(n_clusters, len(X)),
            random_state=42,
            n_init=10,
            max_iter=300
        )
        kmeans.fit(X)
        
        # Asignar clusters
        df['cluster'] = kmeans.labels_
        
        # Métricas de evaluación
        silhouette = silhouette_score(X, kmeans.labels_) if len(set(kmeans.labels_)) > 1 else 0
        calinski = calinski_harabasz_score(X, kmeans.labels_) if len(set(kmeans.labels_)) > 1 else 0
        davies_bouldin = davies_bouldin_score(X, kmeans.labels_) if len(set(kmeans.labels_)) > 1 else 0
        
        # Análisis de clusters
        perfiles = {}
        for i in range(min(n_clusters, len(set(kmeans.labels_)))):
            cluster_data = df[df['cluster'] == i]
            if len(cluster_data) > 0:
                perfiles[i] = {
                    'tamaño': len(cluster_data),
                    'porcentaje': len(cluster_data) / len(df) * 100,
                    'edad_promedio': cluster_data['edad'].mean() if 'edad' in cluster_data.columns else 50,
                    'consultas_promedio': cluster_data['total_consultas'].mean() if 'total_consultas' in cluster_data.columns else 5,
                    'imc_promedio': cluster_data['imc_promedio'].mean() if 'imc_promedio' in cluster_data.columns else 25,
                    'duracion_promedio': cluster_data['duracion_promedio'].mean() if 'duracion_promedio' in cluster_data.columns else 30,
                    'genero_dominante': cluster_data['genero'].mode().iloc[0] if 'genero' in cluster_data.columns and not cluster_data.empty else 'N/A'
                }
        
        resultado = {
            'modelo': kmeans,
            'n_clusters': n_clusters,
            'labels': kmeans.labels_,
            'silhouette_score': silhouette,
            'calinski_score': calinski,
            'davies_bouldin_score': davies_bouldin,
            'perfiles': perfiles,
            'df': df,
            'X': X
        }
        
        self.modelos['kmeans'] = kmeans
        self.resultado_kmeans = resultado
        
        logger.info(f"K-Means entrenado. Silhouette: {silhouette:.4f}")
        return resultado
    
    def reducir_dimensionalidad(self, n_componentes=2):
        """
        Aplicar PCA para reducción de dimensionalidad
        """
        logger.info(f"Aplicando PCA con {n_componentes} componentes...")
        
        X, df = self.preparar_datos_clustering()
        
        if X is None or len(X) < n_componentes:
            logger.warning("No hay suficientes datos para PCA, usando datos sintéticos")
            X, df = self._crear_datos_sinteticos()
        
        pca = PCA(n_components=min(n_componentes, X.shape[1], X.shape[0]-1), random_state=42)
        X_pca = pca.fit_transform(X)
        
        # Explicación de varianza
        varianza_explicada = pca.explained_variance_ratio_
        varianza_acumulada = np.cumsum(varianza_explicada)
        
        resultado = {
            'pca': pca,
            'X_pca': X_pca,
            'varianza_explicada': varianza_explicada,
            'varianza_acumulada': varianza_acumulada,
            'n_componentes': n_componentes
        }
        
        self.resultado_pca = resultado  # <-- Inicializar correctamente
        
        logger.info(f"PCA aplicado. Varianza explicada: {varianza_acumulada[-1]:.4f}")
        return resultado
    
    def optimizar_clusters(self, min_k=2, max_k=10):
        """
        Encontrar el número óptimo de clusters usando el método del codo
        """
        logger.info(f"Optimizando clusters entre {min_k} y {max_k}...")
        
        X, df = self.preparar_datos_clustering()
        
        if X is None or len(X) < max_k:
            logger.warning("No hay suficientes datos, usando datos sintéticos")
            X, df = self._crear_datos_sinteticos()
        
        inertias = []
        silhouette_scores = []
        max_k = min(max_k, len(X))
        
        for k in range(min_k, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
            if len(set(kmeans.labels_)) > 1:
                silhouette_scores.append(silhouette_score(X, kmeans.labels_))
            else:
                silhouette_scores.append(0)
        
        # Encontrar k óptimo (máximo silhouette)
        k_optimo = range(min_k, max_k + 1)[np.argmax(silhouette_scores)] if silhouette_scores else min_k
        
        resultado = {
            'k_values': list(range(min_k, max_k + 1)),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'k_optimo': k_optimo
        }
        
        self.optimizacion_clusters = resultado
        
        logger.info(f"K óptimo encontrado: {k_optimo}")
        return resultado
    
    # ============ GUARDAR Y CARGAR MODELOS ============
    
    def guardar_modelos(self, directorio='ml/modelos'):
        """Guardar modelos entrenados"""
        os.makedirs(directorio, exist_ok=True)
        
        if 'kmeans' in self.modelos:
            joblib.dump(self.modelos['kmeans'], f'{directorio}/kmeans.pkl')
            logger.info(f"K-Means guardado en {directorio}/kmeans.pkl")
        
        if self.resultado_pca is not None and 'pca' in self.resultado_pca:
            joblib.dump(self.resultado_pca['pca'], f'{directorio}/pca.pkl')
            logger.info(f"PCA guardado en {directorio}/pca.pkl")
        
        for nombre, scaler in self.scalers.items():
            joblib.dump(scaler, f'{directorio}/scaler_{nombre}.pkl')
            logger.info(f"Scaler {nombre} guardado en {directorio}/scaler_{nombre}.pkl")
        
        logger.info(f"Modelos guardados en {directorio}/")
    
    def cargar_modelos(self, directorio='ml/modelos'):
        """Cargar modelos guardados"""
        try:
            self.modelos['kmeans'] = joblib.load(f'{directorio}/kmeans.pkl')
            logger.info("Modelo K-Means cargado")
        except:
            logger.warning("No se pudo cargar K-Means")
        
        try:
            pca = joblib.load(f'{directorio}/pca.pkl')
            self.resultado_pca = {'pca': pca}
            logger.info("Modelo PCA cargado")
        except:
            logger.warning("No se pudo cargar PCA")
    
    # ============ UTILIDADES ============
    
    def obtener_segmentacion_pacientes(self):
        """
        Obtener segmentación de pacientes con descripciones
        """
        if self.resultado_kmeans is None:
            self.entrenar_kmeans()
        
        if self.resultado_kmeans is None:
            return {}
        
        perfiles = self.resultado_kmeans['perfiles']
        
        descripciones = {}
        for i, perfil in perfiles.items():
            if perfil['porcentaje'] > 30:
                nombre = f"Paciente Estándar ({perfil['porcentaje']:.0f}%)"
                color = '#4caf50'
            elif perfil['porcentaje'] > 20:
                nombre = f"Paciente de Seguimiento ({perfil['porcentaje']:.0f}%)"
                color = '#ff9800'
            else:
                nombre = f"Paciente Crónico ({perfil['porcentaje']:.0f}%)"
                color = '#f44336'
            
            descripciones[i] = {
                'nombre': nombre,
                'color': color,
                'edad': f"{perfil['edad_promedio']:.0f} años",
                'consultas': f"{perfil['consultas_promedio']:.1f} consultas",
                'imc': f"{perfil['imc_promedio']:.1f}",
                'genero': perfil['genero_dominante'],
                'porcentaje': f"{perfil['porcentaje']:.0f}%"
            }
        
        return descripciones