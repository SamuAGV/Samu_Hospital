# modules/data_cleaner.py - Limpieza y Preprocesamiento de Datos
import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Clase para limpieza y preprocesamiento de datos hospitalarios
    Unidad II: Preparación de los datos
    """
    
    def __init__(self, db):
        self.db = db
        self.df_pacientes = None
        self.df_consultas = None
        self.df_hospitalizaciones = None
        self.df_citas = None
    
    def cargar_datos(self):
        """Cargar todos los datos desde la base de datos"""
        logger.info("Cargando datos desde Supabase...")
        
        # Cargar pacientes
        pacientes = self.db.fetch_all("SELECT * FROM pacientes WHERE activo = true")
        self.df_pacientes = pd.DataFrame(pacientes) if pacientes else pd.DataFrame()
        
        # Cargar consultas
        consultas = self.db.fetch_all("""
            SELECT c.*, p.nombre as paciente_nombre, p.apellido as paciente_apellido,
                   m.nombre as medico_nombre, m.apellido as medico_apellido,
                   e.nombre as especialidad
            FROM consultas c
            JOIN pacientes p ON c.id_paciente = p.id_paciente
            JOIN medicos m ON c.id_medico = m.id_medico
            JOIN especialidades e ON m.id_especialidad = e.id_especialidad
        """)
        self.df_consultas = pd.DataFrame(consultas) if consultas else pd.DataFrame()
        
        # Cargar hospitalizaciones
        hospitalizaciones = self.db.fetch_all("""
            SELECT h.*, p.nombre as paciente_nombre, p.apellido as paciente_apellido
            FROM hospitalizaciones h
            JOIN pacientes p ON h.id_paciente = p.id_paciente
        """)
        self.df_hospitalizaciones = pd.DataFrame(hospitalizaciones) if hospitalizaciones else pd.DataFrame()
        
        # Cargar citas
        citas = self.db.fetch_all("""
            SELECT c.*, p.nombre as paciente_nombre, p.apellido as paciente_apellido,
                   m.nombre as medico_nombre, m.apellido as medico_apellido
            FROM citas c
            JOIN pacientes p ON c.id_paciente = p.id_paciente
            JOIN medicos m ON c.id_medico = m.id_medico
        """)
        self.df_citas = pd.DataFrame(citas) if citas else pd.DataFrame()
        
        logger.info(f"Datos cargados: {len(self.df_pacientes)} pacientes, {len(self.df_consultas)} consultas")
        return self
    
    def limpiar_pacientes(self):
        """Limpiar datos de pacientes"""
        logger.info("Limpiando datos de pacientes...")
        
        if self.df_pacientes.empty:
            return self.df_pacientes
        
        df = self.df_pacientes.copy()
        
        # 1. Eliminar duplicados por email y teléfono
        df = df.drop_duplicates(subset=['email'], keep='first')
        df = df.drop_duplicates(subset=['telefono'], keep='first')
        
        # 2. Manejar valores nulos
        df['nombre'] = df['nombre'].fillna('Desconocido')
        df['apellido'] = df['apellido'].fillna('Desconocido')
        df['telefono'] = df['telefono'].fillna('No registrado')
        df['email'] = df['email'].fillna('no@email.com')
        df['genero'] = df['genero'].fillna('No especificado')
        df['tipo_sangre'] = df['tipo_sangre'].fillna('Desconocido')
        df['alergias'] = df['alergias'].fillna('Ninguna')
        df['enfermedades_cronicas'] = df['enfermedades_cronicas'].fillna('Ninguna')
        
        # 3. Validar y formatear teléfonos
        df['telefono'] = df['telefono'].apply(lambda x: self._formatear_telefono(str(x)) if pd.notna(x) else x)
        
        # 4. Validar emails
        df['email_valido'] = df['email'].apply(lambda x: self._validar_email(str(x)) if pd.notna(x) else False)
        
        # 5. Calcular edad
        df['fecha_nacimiento'] = pd.to_datetime(df['fecha_nacimiento'])
        df['edad'] = (datetime.now() - df['fecha_nacimiento']).dt.days // 365
        
        # 6. Clasificar por edad
        df['grupo_edad'] = pd.cut(
            df['edad'],
            bins=[0, 12, 18, 40, 65, 100],
            labels=['Niño', 'Adolescente', 'Adulto Joven', 'Adulto', 'Adulto Mayor']
        )
        
        # 7. Normalizar nombres
        df['nombre'] = df['nombre'].str.title()
        df['apellido'] = df['apellido'].str.title()
        
        self.df_pacientes_limpio = df
        logger.info(f"Pacientes limpios: {len(df)} registros")
        return df
    
    def limpiar_consultas(self):
        """Limpiar datos de consultas"""
        logger.info("Limpiando datos de consultas...")
        
        if self.df_consultas.empty:
            return self.df_consultas
        
        df = self.df_consultas.copy()
        
        # 1. Convertir fechas
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
        df['fecha'] = df['fecha_hora'].dt.date
        df['hora'] = df['fecha_hora'].dt.hour
        
        # 2. Manejar valores nulos en datos clínicos
        df['peso'] = df['peso'].fillna(df['peso'].median())
        df['altura'] = df['altura'].fillna(df['altura'].median())
        df['presion_arterial'] = df['presion_arterial'].fillna('No registrada')
        df['temperatura'] = df['temperatura'].fillna(36.5)
        df['frecuencia_cardiaca'] = df['frecuencia_cardiaca'].fillna(75)
        
        # 3. Calcular IMC
        df['imc'] = df['peso'] / ((df['altura']/100) ** 2)
        
        # 4. Clasificar IMC
        df['categoria_imc'] = pd.cut(
            df['imc'],
            bins=[0, 18.5, 25, 30, 100],
            labels=['Bajo peso', 'Normal', 'Sobrepeso', 'Obesidad']
        )
        
        # 5. Clasificar presión arterial
        def clasificar_presion(presion):
            if presion == 'No registrada':
                return 'No registrada'
            try:
                sistolica, diastolica = map(int, presion.split('/'))
                if sistolica < 120 and diastolica < 80:
                    return 'Normal'
                elif sistolica < 130 and diastolica < 80:
                    return 'Elevada'
                elif sistolica < 140 or diastolica < 90:
                    return 'Hipertensión Etapa 1'
                else:
                    return 'Hipertensión Etapa 2'
            except:
                return 'No clasificada'
        
        df['categoria_presion'] = df['presion_arterial'].apply(clasificar_presion)
        
        # 6. Calcular duración promedio por especialidad
        if 'duracion_atencion' in df.columns and 'especialidad' in df.columns:
            df['duracion_atencion'] = df['duracion_atencion'].fillna(
                df.groupby('especialidad')['duracion_atencion'].transform('median')
            )
        
        self.df_consultas_limpio = df
        logger.info(f"Consultas limpias: {len(df)} registros")
        return df
    
    def limpiar_hospitalizaciones(self):
        """Limpiar datos de hospitalizaciones"""
        logger.info("Limpiando datos de hospitalizaciones...")
        
        if self.df_hospitalizaciones.empty:
            return self.df_hospitalizaciones
        
        df = self.df_hospitalizaciones.copy()
        
        # 1. Convertir fechas
        df['fecha_ingreso'] = pd.to_datetime(df['fecha_ingreso'])
        if 'fecha_alta' in df.columns:
            df['fecha_alta'] = pd.to_datetime(df['fecha_alta'])
        
        # 2. Calcular días de estancia
        df['dias_estancia'] = (df['fecha_alta'] - df['fecha_ingreso']).dt.days
        
        # 3. Manejar valores nulos
        df['habitacion'] = df['habitacion'].fillna('No asignada')
        df['cama'] = df['cama'].fillna('No asignada')
        df['observaciones'] = df['observaciones'].fillna('Sin observaciones')
        
        # 4. Clasificar estancia
        df['categoria_estancia'] = pd.cut(
            df['dias_estancia'],
            bins=[-1, 3, 7, 14, 100],
            labels=['Corta (1-3 días)', 'Media (4-7 días)', 'Larga (8-14 días)', 'Muy Larga (>14 días)']
        )
        
        self.df_hospitalizaciones_limpio = df
        logger.info(f"Hospitalizaciones limpias: {len(df)} registros")
        return df
    
    def limpiar_citas(self):
        """Limpiar datos de citas"""
        logger.info("Limpiando datos de citas...")
        
        if self.df_citas.empty:
            return self.df_citas
        
        df = self.df_citas.copy()
        
        # 1. Convertir fechas
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'])
        df['fecha'] = df['fecha_hora'].dt.date
        df['hora'] = df['fecha_hora'].dt.hour
        df['dia_semana'] = df['fecha_hora'].dt.day_name()
        
        # 2. Manejar valores nulos
        df['motivo'] = df['motivo'].fillna('Sin motivo especificado')
        df['estado'] = df['estado'].fillna('Programada')
        
        self.df_citas_limpio = df
        logger.info(f"Citas limpias: {len(df)} registros")
        return df
    
    def limpiar_todos(self):
        """Ejecutar limpieza de todos los datasets"""
        self.limpiar_pacientes()
        self.limpiar_consultas()
        self.limpiar_hospitalizaciones()
        self.limpiar_citas()
        return self
    
    def obtener_reportes_limpieza(self):
        """Generar reporte de limpieza"""
        reporte = {
            'pacientes': {
                'original': len(self.df_pacientes),
                'limpio': len(self.df_pacientes_limpio) if hasattr(self, 'df_pacientes_limpio') else 0,
                'duplicados_eliminados': len(self.df_pacientes) - len(self.df_pacientes_limpio) if hasattr(self, 'df_pacientes_limpio') else 0
            },
            'consultas': {
                'original': len(self.df_consultas),
                'limpio': len(self.df_consultas_limpio) if hasattr(self, 'df_consultas_limpio') else 0
            },
            'hospitalizaciones': {
                'original': len(self.df_hospitalizaciones),
                'limpio': len(self.df_hospitalizaciones_limpio) if hasattr(self, 'df_hospitalizaciones_limpio') else 0
            },
            'citas': {
                'original': len(self.df_citas),
                'limpio': len(self.df_citas_limpio) if hasattr(self, 'df_citas_limpio') else 0
            }
        }
        return reporte
    
    def guardar_datos_limpios(self, directorio='data'):
        """Guardar datos limpios en archivos CSV"""
        import os
        os.makedirs(directorio, exist_ok=True)
        
        if hasattr(self, 'df_pacientes_limpio'):
            self.df_pacientes_limpio.to_csv(f'{directorio}/pacientes_limpios.csv', index=False)
        if hasattr(self, 'df_consultas_limpio'):
            self.df_consultas_limpio.to_csv(f'{directorio}/consultas_limpias.csv', index=False)
        if hasattr(self, 'df_hospitalizaciones_limpio'):
            self.df_hospitalizaciones_limpio.to_csv(f'{directorio}/hospitalizaciones_limpias.csv', index=False)
        if hasattr(self, 'df_citas_limpio'):
            self.df_citas_limpio.to_csv(f'{directorio}/citas_limpias.csv', index=False)
        
        logger.info(f"Datos guardados en {directorio}/")
    
    # ============ MÉTODOS PRIVADOS ============
    
    def _formatear_telefono(self, telefono):
        """Formatear número de teléfono"""
        telefono = re.sub(r'[^0-9]', '', telefono)
        if len(telefono) == 10:
            return f"{telefono[:3]}-{telefono[3:6]}-{telefono[6:]}"
        return telefono
    
    def _validar_email(self, email):
        """Validar formato de email"""
        patron = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(patron, email))