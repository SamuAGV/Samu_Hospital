class PacienteModule:
    def __init__(self, db):
        self.db = db
    
    def listar_todos(self):
        query = "SELECT * FROM pacientes WHERE activo = true ORDER BY apellido, nombre"
        return self.db.fetch_all(query)
    
    def crear_paciente(self, datos):
        query = """
        INSERT INTO pacientes (nombre, apellido, fecha_nacimiento, genero, 
                               telefono, email, direccion, tipo_sangre, alergias)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id_paciente
        """
        return self.db.execute(query, datos)
    
    def obtener_paciente(self, id_paciente):
        query = "SELECT * FROM pacientes WHERE id_paciente = %s AND activo = true"
        return self.db.fetch_one(query, (id_paciente,))
    
    def buscar_pacientes(self, criterio):
        query = """
        SELECT * FROM pacientes 
        WHERE activo = true 
        AND (nombre ILIKE %s OR apellido ILIKE %s OR telefono ILIKE %s)
        """
        return self.db.fetch_all(query, (f'%{criterio}%', f'%{criterio}%', f'%{criterio}%'))
    
    def historial_paciente(self, id_paciente):
        consultas = """
        SELECT c.*, e.nombre as especialidad, m.nombre as medico_nombre
        FROM consultas c
        JOIN medicos m ON c.id_medico = m.id_medico
        JOIN especialidades e ON m.id_especialidad = e.id_especialidad
        WHERE c.id_paciente = %s
        ORDER BY c.fecha_hora DESC
        """
        hospitalizaciones = """
        SELECT * FROM hospitalizaciones 
        WHERE id_paciente = %s 
        ORDER BY fecha_ingreso DESC
        """
        return {
            'consultas': self.db.fetch_all(consultas, (id_paciente,)),
            'hospitalizaciones': self.db.fetch_all(hospitalizaciones, (id_paciente,))
        }
