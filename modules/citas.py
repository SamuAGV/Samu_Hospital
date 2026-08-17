class CitaModule:
    def __init__(self, db):
        self.db = db
    
    def listar_citas(self, fecha=None, id_medico=None, estado=None):
        query = """
        SELECT c.*, p.nombre as paciente_nombre, p.apellido as paciente_apellido,
               m.nombre as medico_nombre, m.apellido as medico_apellido
        FROM citas c
        JOIN pacientes p ON c.id_paciente = p.id_paciente
        JOIN medicos m ON c.id_medico = m.id_medico
        WHERE 1=1
        """
        params = []
        if fecha:
            query += " AND DATE(c.fecha_hora) = %s"
            params.append(fecha)
        if id_medico:
            query += " AND c.id_medico = %s"
            params.append(id_medico)
        if estado:
            query += " AND c.estado = %s"
            params.append(estado)
        query += " ORDER BY c.fecha_hora DESC"
        return self.db.fetch_all(query, tuple(params) if params else None)
    
    def agendar_cita(self, datos):
        query = """
        INSERT INTO citas (id_paciente, id_medico, id_consultorio, fecha_hora, 
                          duracion, estado, motivo)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id_cita
        """
        return self.db.execute(query, (
            datos['id_paciente'], datos['id_medico'], datos.get('id_consultorio'),
            datos['fecha_hora'], datos.get('duracion', 30), datos.get('estado', 'Programada'),
            datos.get('motivo')
        ))
    
    def cancelar_cita(self, id_cita, motivo):
        query = """
        UPDATE citas 
        SET estado = 'Cancelada', fecha_cancelacion = CURRENT_TIMESTAMP,
            motivo_cancelacion = %s
        WHERE id_cita = %s
        """
        return self.db.execute(query, (motivo, id_cita))
