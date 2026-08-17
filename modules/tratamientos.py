class TratamientoModule:
    def __init__(self, db):
        self.db = db
    
    def prescribir_tratamiento(self, datos):
        query = """
        INSERT INTO tratamientos (id_consulta, id_medicamento, dosis, frecuencia,
                                 duracion_dias, indicaciones, fecha_inicio, fecha_fin)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id_tratamiento
        """
        return self.db.execute(query, (
            datos['id_consulta'], datos['id_medicamento'], datos.get('dosis'),
            datos.get('frecuencia'), datos.get('duracion_dias'), datos.get('indicaciones'),
            datos.get('fecha_inicio'), datos.get('fecha_fin')
        ))
    
    def obtener_medicamentos_mas_usados(self, fecha_inicio, fecha_fin):
        query = """
        SELECT m.nombre, m.principio_activo, COUNT(*) as frecuencia
        FROM tratamientos t
        JOIN medicamentos m ON t.id_medicamento = m.id_medicamento
        JOIN consultas c ON t.id_consulta = c.id_consulta
        WHERE c.fecha_hora BETWEEN %s AND %s
        GROUP BY m.id_medicamento, m.nombre, m.principio_activo
        ORDER BY frecuencia DESC
        LIMIT 10
        """
        return self.db.fetch_all(query, (fecha_inicio, fecha_fin))
