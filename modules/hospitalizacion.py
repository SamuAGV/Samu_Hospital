# modules/hospitalizacion.py - Módulo de Hospitalización
class HospitalizacionModule:
    def __init__(self, db):
        self.db = db
    
    def ingresar_paciente(self, datos):
        query = """
        INSERT INTO hospitalizaciones (id_paciente, id_medico_responsable,
                                      habitacion, cama, motivo_ingreso,
                                      diagnostico_ingreso, tipo_ingreso)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id_hospitalizacion
        """
        return self.db.execute(query, datos)
    
    def dar_alta(self, id_hospitalizacion, notas):
        query = """
        UPDATE hospitalizaciones 
        SET fecha_alta = CURRENT_TIMESTAMP, estado = 'Alta',
            observaciones = COALESCE(observaciones, '') || %s
        WHERE id_hospitalizacion = %s
        RETURNING id_paciente
        """
        return self.db.execute(query, (notas, id_hospitalizacion))
    
    def obtener_ocupacion_hospitalaria(self):
        query = """
        SELECT 
            COALESCE(COUNT(*), 0) as total_camas,
            COALESCE(SUM(CASE WHEN estado = 'Activa' THEN 1 ELSE 0 END), 0) as camas_ocupadas,
            COALESCE(ROUND(100.0 * SUM(CASE WHEN estado = 'Activa' THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0), 2), 0) as porcentaje_ocupacion
        FROM hospitalizaciones
        WHERE estado IN ('Activa', 'Alta')
        """
        result = self.db.fetch_one(query)
        if result is None:
            return {
                'total_camas': 0,
                'camas_ocupadas': 0,
                'porcentaje_ocupacion': 0
            }
        return result
    
    def obtener_estancia_promedio(self, fecha_inicio, fecha_fin):
        query = """
        SELECT COALESCE(AVG(EXTRACT(EPOCH FROM (fecha_alta - fecha_ingreso)) / 86400), 0) as dias_promedio
        FROM hospitalizaciones
        WHERE fecha_ingreso BETWEEN %s AND %s AND estado = 'Alta'
        """
        result = self.db.fetch_one(query, (fecha_inicio, fecha_fin))
        if result is None:
            return {'dias_promedio': 0}
        return result
    
    def listar_ingresos_activos(self):
        query = """
        SELECT h.*, 
               p.nombre as paciente_nombre, 
               p.apellido as paciente_apellido,
               m.nombre as medico_nombre,
               m.apellido as medico_apellido
        FROM hospitalizaciones h
        JOIN pacientes p ON h.id_paciente = p.id_paciente
        JOIN medicos m ON h.id_medico_responsable = m.id_medico
        WHERE h.estado = 'Activa'
        ORDER BY h.fecha_ingreso DESC
        """
        return self.db.fetch_all(query)
