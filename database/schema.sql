-- ============================================================
-- MEDINSIGHT-HOSPITAL - Esquema de Base de Datos
-- Supabase PostgreSQL
-- ============================================================

-- 1. TABLA PACIENTES
CREATE TABLE IF NOT EXISTS pacientes (
    id_paciente SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    apellido VARCHAR(100) NOT NULL,
    fecha_nacimiento DATE NOT NULL,
    genero VARCHAR(20),
    telefono VARCHAR(20),
    email VARCHAR(100),
    direccion TEXT,
    tipo_sangre VARCHAR(5),
    alergias TEXT,
    enfermedades_cronicas TEXT,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    activo BOOLEAN DEFAULT TRUE
);

-- 2. TABLA ESPECIALIDADES
CREATE TABLE IF NOT EXISTS especialidades (
    id_especialidad SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL UNIQUE,
    descripcion TEXT,
    costo_consulta DECIMAL(10,2) DEFAULT 0,
    tiempo_promedio_atencion INT DEFAULT 30
);

-- 3. TABLA MEDICOS
CREATE TABLE IF NOT EXISTS medicos (
    id_medico SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    apellido VARCHAR(100) NOT NULL,
    id_especialidad INT REFERENCES especialidades(id_especialidad),
    cedula_profesional VARCHAR(50) UNIQUE NOT NULL,
    telefono VARCHAR(20),
    email VARCHAR(100),
    fecha_contratacion DATE,
    salario DECIMAL(10,2),
    activo BOOLEAN DEFAULT TRUE
);

-- 4. TABLA CONSULTORIOS
CREATE TABLE IF NOT EXISTS consultorios (
    id_consultorio SERIAL PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    ubicacion VARCHAR(100),
    id_especialidad INT REFERENCES especialidades(id_especialidad),
    activo BOOLEAN DEFAULT TRUE
);

-- 5. TABLA CITAS
CREATE TABLE IF NOT EXISTS citas (
    id_cita SERIAL PRIMARY KEY,
    id_paciente INT REFERENCES pacientes(id_paciente),
    id_medico INT REFERENCES medicos(id_medico),
    id_consultorio INT REFERENCES consultorios(id_consultorio),
    fecha_hora TIMESTAMP NOT NULL,
    duracion INT DEFAULT 30,
    estado VARCHAR(30) CHECK (estado IN ('Programada', 'Atendida', 'Cancelada', 'Reprogramada')),
    motivo TEXT,
    fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_cancelacion TIMESTAMP,
    motivo_cancelacion TEXT
);

-- 6. TABLA CONSULTAS
CREATE TABLE IF NOT EXISTS consultas (
    id_consulta SERIAL PRIMARY KEY,
    id_cita INT REFERENCES citas(id_cita),
    id_paciente INT REFERENCES pacientes(id_paciente),
    id_medico INT REFERENCES medicos(id_medico),
    fecha_hora TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    peso DECIMAL(5,2),
    altura DECIMAL(5,2),
    presion_arterial VARCHAR(20),
    temperatura DECIMAL(4,1),
    frecuencia_cardiaca INT,
    sintomas TEXT,
    notas_clinicas TEXT,
    duracion_atencion INT,
    tipo_consulta VARCHAR(50) CHECK (tipo_consulta IN ('Primera vez', 'Seguimiento', 'Urgencia'))
);

-- 7. TABLA DIAGNOSTICOS
CREATE TABLE IF NOT EXISTS diagnosticos (
    id_diagnostico SERIAL PRIMARY KEY,
    id_consulta INT REFERENCES consultas(id_consulta),
    codigo_cie10 VARCHAR(10) NOT NULL,
    nombre VARCHAR(200) NOT NULL,
    descripcion TEXT,
    tipo VARCHAR(30) CHECK (tipo IN ('Principal', 'Secundario', 'Comorbilidad')),
    fecha_diagnostico TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 8. TABLA MEDICAMENTOS
CREATE TABLE IF NOT EXISTS medicamentos (
    id_medicamento SERIAL PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    principio_activo VARCHAR(100),
    presentacion VARCHAR(50),
    concentracion VARCHAR(50),
    precio_unitario DECIMAL(10,2),
    requiere_receta BOOLEAN DEFAULT TRUE,
    stock INT DEFAULT 0,
    stock_minimo INT DEFAULT 10
);

-- 9. TABLA TRATAMIENTOS
CREATE TABLE IF NOT EXISTS tratamientos (
    id_tratamiento SERIAL PRIMARY KEY,
    id_consulta INT REFERENCES consultas(id_consulta),
    id_medicamento INT REFERENCES medicamentos(id_medicamento),
    dosis VARCHAR(50),
    frecuencia VARCHAR(50),
    duracion_dias INT,
    indicaciones TEXT,
    fecha_inicio DATE,
    fecha_fin DATE,
    activo BOOLEAN DEFAULT TRUE
);

-- 10. TABLA HOSPITALIZACIONES
CREATE TABLE IF NOT EXISTS hospitalizaciones (
    id_hospitalizacion SERIAL PRIMARY KEY,
    id_paciente INT REFERENCES pacientes(id_paciente),
    id_medico_responsable INT REFERENCES medicos(id_medico),
    fecha_ingreso TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_alta TIMESTAMP,
    habitacion VARCHAR(20),
    cama VARCHAR(10),
    motivo_ingreso TEXT,
    diagnostico_ingreso TEXT,
    estado VARCHAR(30) CHECK (estado IN ('Activa', 'Alta', 'Traslado', 'Fallecimiento')),
    tipo_ingreso VARCHAR(30) CHECK (tipo_ingreso IN ('Programado', 'Urgencia')),
    observaciones TEXT
);

-- 11. TABLA ESTUDIOS_CLINICOS
CREATE TABLE IF NOT EXISTS estudios_clinicos (
    id_estudio SERIAL PRIMARY KEY,
    id_consulta INT REFERENCES consultas(id_consulta),
    id_paciente INT REFERENCES pacientes(id_paciente),
    nombre_estudio VARCHAR(100) NOT NULL,
    tipo_estudio VARCHAR(50) CHECK (tipo_estudio IN ('Laboratorio', 'Imagen', 'Patología', 'Genético')),
    fecha_solicitud TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    fecha_resultado TIMESTAMP,
    resultado TEXT,
    archivo_url TEXT,
    estado VARCHAR(30) CHECK (estado IN ('Solicitado', 'En Proceso', 'Completado', 'Cancelado'))
);

-- 12. TABLA PAGOS
CREATE TABLE IF NOT EXISTS pagos (
    id_pago SERIAL PRIMARY KEY,
    id_paciente INT REFERENCES pacientes(id_paciente),
    id_consulta INT REFERENCES consultas(id_consulta),
    id_hospitalizacion INT REFERENCES hospitalizaciones(id_hospitalizacion),
    monto_total DECIMAL(10,2) NOT NULL,
    monto_pagado DECIMAL(10,2) DEFAULT 0,
    metodo_pago VARCHAR(30) CHECK (metodo_pago IN ('Efectivo', 'Tarjeta', 'Transferencia', 'Seguro')),
    estado_pago VARCHAR(30) CHECK (estado_pago IN ('Pendiente', 'Parcial', 'Pagado', 'Cancelado')),
    fecha_pago TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    comprobante_url TEXT
);

-- 13. ÍNDICES
CREATE INDEX IF NOT EXISTS idx_citas_fecha_hora ON citas(fecha_hora);
CREATE INDEX IF NOT EXISTS idx_citas_estado ON citas(estado);
CREATE INDEX IF NOT EXISTS idx_citas_paciente ON citas(id_paciente);
CREATE INDEX IF NOT EXISTS idx_consultas_fecha ON consultas(fecha_hora);
CREATE INDEX IF NOT EXISTS idx_hospitalizaciones_estado ON hospitalizaciones(estado);
CREATE INDEX IF NOT EXISTS idx_diagnosticos_codigo ON diagnosticos(codigo_cie10);
CREATE INDEX IF NOT EXISTS idx_pagos_fecha ON pagos(fecha_pago);

-- 14. DATOS INICIALES - ESPECIALIDADES
INSERT INTO especialidades (nombre, descripcion, costo_consulta) VALUES
('Medicina General', 'Atención primaria y consultas generales', 350.00),
('Cardiología', 'Diagnóstico y tratamiento de enfermedades del corazón', 650.00),
('Pediatría', 'Atención médica para niños y adolescentes', 400.00),
('Ginecología', 'Salud de la mujer y sistema reproductor', 550.00),
('Traumatología', 'Lesiones del sistema musculoesquelético', 500.00),
('Neurología', 'Trastornos del sistema nervioso', 700.00),
('Dermatología', 'Enfermedades de la piel', 450.00)
ON CONFLICT (nombre) DO NOTHING;

-- 15. DATOS INICIALES - MEDICAMENTOS
INSERT INTO medicamentos (nombre, principio_activo, presentacion, precio_unitario) VALUES
('Paracetamol', 'Acetaminofén', 'Tableta 500mg', 15.50),
('Ibuprofeno', 'Ibuprofeno', 'Tableta 400mg', 22.30),
('Amoxicilina', 'Amoxicilina', 'Cápsula 500mg', 35.00),
('Loratadina', 'Loratadina', 'Tableta 10mg', 12.80),
('Omeprazol', 'Omeprazol', 'Cápsula 20mg', 28.50)
ON CONFLICT DO NOTHING;