class MedicoModule:
    def __init__(self, db):
        self.db = db
    
    def listar_todos(self):
        query = """
        SELECT m.*, e.nombre as especialidad 
        FROM medicos m
        JOIN especialidades e ON m.id_especialidad = e.id_especialidad
        WHERE m.activo = true
        """
        return self.db.fetch_all(query)
    
    def crear_medico(self, datos):
        query = """
        INSERT INTO medicos (nombre, apellido, id_especialidad, cedula_profesional,
                            telefono, email, fecha_contratacion, salario)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id_medico
        """
        return self.db.execute(query, datos)
