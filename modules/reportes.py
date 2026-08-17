class ReporteModule:
    def __init__(self, db):
        self.db = db
    
    def obtener_resumen_general(self, fecha_inicio, fecha_fin):
        query = """
        SELECT 
            (SELECT COUNT(*) FROM pacientes WHERE activo = true) as total_pacientes,
            (SELECT COUNT(*) FROM consultas WHERE fecha_hora BETWEEN %s AND %s) as total_consultas,
            (SELECT COALESCE(SUM(monto_total), 0) FROM pagos WHERE fecha_pago BETWEEN %s AND %s) as ingresos_totales
        """
        return self.db.fetch_one(query, (fecha_inicio, fecha_fin, fecha_inicio, fecha_fin))
    
    def analizar_demanda(self, fecha_inicio, fecha_fin):
        query = """
        SELECT DATE(fecha_hora) as fecha, COUNT(*) as total
        FROM consultas
        WHERE fecha_hora BETWEEN %s AND %s
        GROUP BY DATE(fecha_hora)
        ORDER BY fecha
        """
        return self.db.fetch_all(query, (fecha_inicio, fecha_fin))
    
    def analizar_horas_pico(self, fecha_inicio, fecha_fin):
        query = """
        SELECT EXTRACT(HOUR FROM fecha_hora) as hora, COUNT(*) as cantidad
        FROM consultas
        WHERE fecha_hora BETWEEN %s AND %s
        GROUP BY hora
        ORDER BY cantidad DESC
        LIMIT 5
        """
        return self.db.fetch_all(query, (fecha_inicio, fecha_fin))
    
    def calcular_kpis(self, fecha_inicio, fecha_fin):
        return {
            'efectividad_atencion': 0,
            'productividad_medica': 0,
            'satisfaccion_estimada': 0,
            'tasa_ocupacion': 0,
            'tasa_utilizacion_camas': 0
        }
