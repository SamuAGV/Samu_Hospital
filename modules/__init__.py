from .database import Database
from .pacientes import PacienteModule
from .medicos import MedicoModule
from .citas import CitaModule
from .consultas import ConsultaModule
from .diagnosticos import DiagnosticoModule
from .tratamientos import TratamientoModule
from .hospitalizacion import HospitalizacionModule
from .reportes import ReporteModule

__all__ = [
    'Database',
    'PacienteModule',
    'MedicosModule',
    'CitaModule',
    'ConsultaModule',
    'DiagnosticoModule',
    'TratamientoModule',
    'HospitalizacionModule',
    'ReporteModule'
]
