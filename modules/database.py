# modules/database.py - Conexión a Supabase
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import streamlit as st
import logging

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.connection = None
        self.cursor = None
        self.connected = False
    
    @st.cache_resource
    def connect(_self):
        """Establecer conexión a Supabase PostgreSQL"""
        try:
            # Primero intentar con variables desglosadas
            host = os.getenv('DB_HOST')
            port = os.getenv('DB_PORT', '5432')
            database = os.getenv('DB_NAME', 'postgres')
            user = os.getenv('DB_USER', 'postgres')
            password = os.getenv('DB_PASSWORD')
            
            if host and password:
                # Usar credenciales desglosadas
                _self.connection = psycopg2.connect(
                    host=host,
                    port=port,
                    database=database,
                    user=user,
                    password=password,
                    sslmode='require',
                    connect_timeout=10
                )
            else:
                # Intentar con DATABASE_URL
                database_url = os.getenv('DATABASE_URL')
                if not database_url:
                    raise Exception("No hay credenciales configuradas en .env")
                _self.connection = psycopg2.connect(database_url, sslmode='require', connect_timeout=10)
            
            _self.cursor = _self.connection.cursor(cursor_factory=RealDictCursor)
            _self.connected = True
            logger.info("✅ Conectado a Supabase")
            return _self.connection
            
        except Exception as e:
            logger.error(f"❌ Error de conexión: {e}")
            st.error(f"❌ Error al conectar a Supabase: {e}")
            st.info("Verifica tus credenciales en el archivo .env")
            _self.connected = False
            return None
    
    def execute(self, query, params=None):
        """Ejecutar query y retornar resultado"""
        if not self.connected:
            self.connect()
            if not self.connected:
                raise Exception("Base de datos no conectada")
        
        try:
            self.cursor.execute(query, params)
            query_upper = query.strip().upper()
            
            if query_upper.startswith('SELECT'):
                return self.cursor.fetchall()
            elif query_upper.startswith('INSERT') or query_upper.startswith('UPDATE') or query_upper.startswith('DELETE'):
                self.connection.commit()
                if query_upper.startswith('INSERT'):
                    return self.cursor.fetchone()
                return {'affected_rows': self.cursor.rowcount}
            else:
                self.connection.commit()
                return {'affected_rows': self.cursor.rowcount}
                
        except Exception as e:
            if self.connection:
                self.connection.rollback()
            logger.error(f"Error en query: {e}")
            raise Exception(f"Error en query: {e}")
    
    def fetch_one(self, query, params=None):
        """Obtener un solo registro"""
        try:
            result = self.execute(query, params)
            return result[0] if result and len(result) > 0 else None
        except Exception as e:
            logger.error(f"Error en fetch_one: {e}")
            return None
    
    def fetch_all(self, query, params=None):
        """Obtener todos los registros"""
        try:
            return self.execute(query, params)
        except Exception as e:
            logger.error(f"Error en fetch_all: {e}")
            return []
    
    def test_connection(self):
        """Probar conexión"""
        try:
            result = self.fetch_one("SELECT version() as version")
            return result['version'] if result else None
        except Exception as e:
            return None
