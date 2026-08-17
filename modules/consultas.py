class ConsultaModule:
    def __init__(self, db):
        self.db = db
    
    def registrar_consulta(self, datos):
        query = """
        INSERT INTO consultas (id_cita, id_paciente, id_medico, fecha_hora,
                              peso, altura, presion_arterial, temperatura,
                              frecuencia_cardiaca, sintomas, notas_clinicas,
                              duracion_atencion, tipo_consulta)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id_consulta
        """
        return self.db.execute(query, datos)
    
    def obtener_consultas_por_fecha(self, fecha_inicio, fecha_fin):
        query = """
        SELECT c.*, p.nombre as paciente_nombre, p.apellido as paciente_apellido,
               m.nombre as medico_nombre, m.apellido as medico_apellido,
               e.nombre as especialidad
        FROM consultas c
        JOIN pacientes p ON c.id_paciente = p.id_paciente
        JOIN medicos m ON c.id_medico = m.id_medico
        JOIN especialidades e ON m.id_especialidad = e.id_especialidad
        WHERE c.fecha_hora BETWEEN %s AND %s
        ORDER BY c.fecha_hora DESC
        """
        return self.db.fetch_all(query, (fecha_inicio, fecha_fin))
