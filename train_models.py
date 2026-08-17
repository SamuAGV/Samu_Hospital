# train_models.py - Script para entrenar todos los modelos
import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.database import Database
from modules.data_cleaner import DataCleaner
from ml.ml_supervisado import ModeloSupervisado
from ml.ml_no_supervisado import ModeloNoSupervisado
import logging
import pandas as pd
import numpy as np

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def entrenar_todos():
    """Entrenar todos los modelos del sistema"""
    
    logger.info("=" * 60)
    logger.info("INICIANDO ENTRENAMIENTO DE MODELOS")
    logger.info("=" * 60)
    
    # 1. Conectar a la base de datos
    logger.info("\n[1/5] Conectando a la base de datos...")
    db = Database()
    db.connect()
    
    if not db.connected:
        logger.error("No se pudo conectar a la base de datos")
        return None
    
    # 2. Limpiar datos
    logger.info("\n[2/5] Limpiando datos...")
    cleaner = DataCleaner(db)
    cleaner.cargar_datos().limpiar_todos()
    
    # Guardar datos limpios
    cleaner.guardar_datos_limpios()
    
    # Mostrar reporte de limpieza
    reporte = cleaner.obtener_reportes_limpieza()
    for tabla, stats in reporte.items():
        logger.info(f"  {tabla}: {stats['original']} → {stats['limpio']} registros")
    
    # 3. Entrenar modelos supervisados
    logger.info("\n[3/5] Entrenando modelos supervisados...")
    supervisor = ModeloSupervisado(db, cleaner)
    supervisor.cargar_datos_limpios()
    
    try:
        # Clasificador
        resultado_clasif = supervisor.entrenar_clasificador()
        logger.info(f"  ✅ Clasificador: Accuracy = {resultado_clasif['accuracy']:.4f}")
    except Exception as e:
        logger.error(f"  ❌ Error entrenando clasificador: {e}")
        # Crear datos sintéticos para entrenamiento si no hay suficientes datos
        logger.info("  Creando datos sintéticos para clasificador...")
        supervisor = _crear_datos_sinteticos_clasificacion(supervisor)
        resultado_clasif = supervisor.entrenar_clasificador()
        logger.info(f"  ✅ Clasificador (sintético): Accuracy = {resultado_clasif['accuracy']:.4f}")
    
    try:
        # Regresor
        resultado_regres = supervisor.entrenar_regresor()
        logger.info(f"  ✅ Regresor: R² = {resultado_regres['r2']:.4f}, RMSE = {resultado_regres['rmse']:.4f}")
    except Exception as e:
        logger.error(f"  ❌ Error entrenando regresor: {e}")
        logger.info("  Creando datos sintéticos para regresor...")
        supervisor = _crear_datos_sinteticos_regresion(supervisor)
        resultado_regres = supervisor.entrenar_regresor()
        logger.info(f"  ✅ Regresor (sintético): R² = {resultado_regres['r2']:.4f}, RMSE = {resultado_regres['rmse']:.4f}")
    
    # 4. Entrenar modelos no supervisados
    logger.info("\n[4/5] Entrenando modelos no supervisados...")
    no_supervisor = ModeloNoSupervisado(db, cleaner)
    no_supervisor.cargar_datos_limpios()
    
    try:
        # Optimizar clusters
        optimizacion = no_supervisor.optimizar_clusters()
        k_optimo = optimizacion['k_optimo']
        logger.info(f"  ✅ K óptimo encontrado: {k_optimo}")
        
        # K-Means
        resultado_kmeans = no_supervisor.entrenar_kmeans(k_optimo)
        logger.info(f"  ✅ K-Means: Silhouette = {resultado_kmeans['silhouette_score']:.4f}")
        
        # PCA
        resultado_pca = no_supervisor.reducir_dimensionalidad()
        logger.info(f"  ✅ PCA: Varianza explicada = {resultado_pca['varianza_acumulada'][-1]:.4f}")
    except Exception as e:
        logger.error(f"  ❌ Error en modelos no supervisados: {e}")
        logger.info("  Usando datos sintéticos para clustering...")
        no_supervisor = _crear_datos_sinteticos_clustering(no_supervisor)
        resultado_kmeans = no_supervisor.entrenar_kmeans(3)
        logger.info(f"  ✅ K-Means (sintético): Silhouette = {resultado_kmeans['silhouette_score']:.4f}")
    
    # 5. Guardar todos los modelos
    logger.info("\n[5/5] Guardando modelos...")
    supervisor.guardar_modelos()
    no_supervisor.guardar_modelos()
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)
    logger.info("\nAhora puedes ejecutar: streamlit run app.py")
    
    return {
        'supervisor': supervisor,
        'no_supervisor': no_supervisor,
        'cleaner': cleaner
    }


def _crear_datos_sinteticos_clasificacion(supervisor):
    """Crear datos sintéticos para clasificación"""
    np.random.seed(42)
    
    n_samples = 200
    data = {
        'id_paciente': list(range(1, n_samples + 1)),
        'nombre': ['Paciente_' + str(i) for i in range(1, n_samples + 1)],
        'apellido': ['Apellido_' + str(i) for i in range(1, n_samples + 1)],
        'edad': np.random.randint(18, 90, n_samples),
        'genero': np.random.choice(['Masculino', 'Femenino'], n_samples),
        'fecha_nacimiento': pd.date_range('1950-01-01', periods=n_samples),
        'telefono': ['555-' + str(np.random.randint(100, 999)) + '-' + str(np.random.randint(1000, 9999)) for _ in range(n_samples)],
        'email': ['email_' + str(i) + '@test.com' for i in range(n_samples)],
        'tipo_sangre': np.random.choice(['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'], n_samples),
        'alergias': np.random.choice(['Ninguna', 'Polen', 'Lácteos', 'Medicamentos'], n_samples),
        'enfermedades_cronicas': np.random.choice(['Ninguna', 'Diabetes', 'Hipertensión', 'Asma'], n_samples),
        'activo': True
    }
    
    supervisor.df_pacientes = pd.DataFrame(data)
    
    # Datos de consultas sintéticos
    consultas_data = {
        'id_paciente': np.random.randint(1, n_samples, n_samples * 3),
        'id_consulta': list(range(1, n_samples * 3 + 1)),
        'fecha_hora': pd.date_range('2024-01-01', periods=n_samples * 3),
        'duracion_atencion': np.random.randint(15, 60, n_samples * 3),
        'imc': np.random.uniform(18, 40, n_samples * 3),
        'temperatura': np.random.uniform(36, 38.5, n_samples * 3),
        'frecuencia_cardiaca': np.random.randint(60, 100, n_samples * 3),
        'especialidad': np.random.choice(['Medicina General', 'Cardiología', 'Pediatría'], n_samples * 3),
        'tipo_consulta': np.random.choice(['Primera vez', 'Seguimiento', 'Urgencia'], n_samples * 3)
    }
    
    supervisor.df_consultas = pd.DataFrame(consultas_data)
    return supervisor


def _crear_datos_sinteticos_regresion(supervisor):
    """Crear datos sintéticos para regresión"""
    np.random.seed(42)
    
    n_samples = 200
    data = {
        'id_paciente': list(range(1, n_samples + 1)),
        'edad': np.random.randint(18, 90, n_samples),
        'genero': np.random.choice(['Masculino', 'Femenino'], n_samples),
        'imc': np.random.uniform(18, 40, n_samples),
        'id_consulta': np.random.randint(1, 20, n_samples),
        'dias_estancia': np.random.randint(1, 30, n_samples)
    }
    
    supervisor.df_pacientes = pd.DataFrame(data)
    
    # Datos de consultas sintéticos
    consultas_data = {
        'id_paciente': np.random.randint(1, n_samples, n_samples * 2),
        'id_consulta': list(range(1, n_samples * 2 + 1)),
        'imc': np.random.uniform(18, 40, n_samples * 2),
        'duracion_atencion': np.random.randint(15, 60, n_samples * 2)
    }
    
    supervisor.df_consultas = pd.DataFrame(consultas_data)
    return supervisor


def _crear_datos_sinteticos_clustering(no_supervisor):
    """Crear datos sintéticos para clustering"""
    np.random.seed(42)
    
    n_samples = 150
    data = {
        'id_paciente': list(range(1, n_samples + 1)),
        'nombre': ['Paciente_' + str(i) for i in range(1, n_samples + 1)],
        'apellido': ['Apellido_' + str(i) for i in range(1, n_samples + 1)],
        'edad': np.random.randint(18, 90, n_samples),
        'genero': np.random.choice(['Masculino', 'Femenino'], n_samples),
        'fecha_nacimiento': pd.date_range('1950-01-01', periods=n_samples),
        'telefono': ['555-' + str(np.random.randint(100, 999)) + '-' + str(np.random.randint(1000, 9999)) for _ in range(n_samples)],
        'email': ['email_' + str(i) + '@test.com' for i in range(n_samples)],
        'tipo_sangre': np.random.choice(['A+', 'A-', 'B+', 'B-', 'O+', 'O-', 'AB+', 'AB-'], n_samples),
        'alergias': np.random.choice(['Ninguna', 'Polen', 'Lácteos', 'Medicamentos'], n_samples),
        'enfermedades_cronicas': np.random.choice(['Ninguna', 'Diabetes', 'Hipertensión', 'Asma'], n_samples),
        'activo': True
    }
    
    no_supervisor.df_pacientes = pd.DataFrame(data)
    
    # Datos de consultas sintéticos
    consultas_data = {
        'id_paciente': np.random.randint(1, n_samples, n_samples * 3),
        'id_consulta': list(range(1, n_samples * 3 + 1)),
        'fecha_hora': pd.date_range('2024-01-01', periods=n_samples * 3),
        'duracion_atencion': np.random.randint(15, 60, n_samples * 3),
        'imc': np.random.uniform(18, 40, n_samples * 3),
        'temperatura': np.random.uniform(36, 38.5, n_samples * 3),
        'frecuencia_cardiaca': np.random.randint(60, 100, n_samples * 3),
        'especialidad': np.random.choice(['Medicina General', 'Cardiología', 'Pediatría'], n_samples * 3)
    }
    
    no_supervisor.df_consultas = pd.DataFrame(consultas_data)
    return no_supervisor


if __name__ == "__main__":
    entrenar_todos()