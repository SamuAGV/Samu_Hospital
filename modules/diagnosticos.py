class DiagnosticoModule:
    def __init__(self, db):
        self.db = db
    
    def registrar_diagnostico(self, datos):
        query = """
        INSERT INTO diagnosticos (id_consulta, codigo_cie10, nombre, descripcion, tipo)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id_diagnostico
        """
        return self.db.execute(query, datos)
    
    def obtener_diagnosticos_frecuentes(self, fecha_inicio, fecha_fin, limite=10):
        query = """
        SELECT d.codigo_cie10, d.nombre, COUNT(*) as frecuencia
        FROM diagnosticos d
        JOIN consultas c ON d.id_consulta = c.id_consulta
        WHERE c.fecha_hora BETWEEN %s AND %s
        GROUP BY d.codigo_cie10, d.nombre
        ORDER BY frecuencia DESC
        LIMIT %s
        """
        return self.db.fetch_all(query, (fecha_inicio, fecha_fin, limite))
