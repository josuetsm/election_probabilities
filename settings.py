import logging
from logging.config import dictConfig
from typing import List, Any, Union

logging_config = dict(
    version=1,
    formatters={
        'f': {'format': '%(Pastime)s %(name)-12s %(levelname)-8s %(message)s'}
    },
    handlers={
        'h': {'class': 'logging.StreamHandler',
              'formatter': 'f',
              'level': logging.DEBUG}
    },
    root={
        'handlers': ['h'],
        'level': logging.DEBUG,
    },
)

dictConfig(logging_config)

logger = logging.getLogger()

headers = {'Escenario_Elecciones': ['cod_elec', 'glosa_elec', 'elec_fecha', 'tipo_zona', 'tipo_eleccion', 'total_electores'],
           'Escenario_Candidatos': ['cod_cand', 'cod_elec', 'cod_zona', 'can_orden', 'glosa_cand', 'cod_part', 'cod_pacto',
                                    'cod_subp', 'cod_ind', 'can_pagina', 'glosa_nombre', 'glosa_apellido', 'cod_genero'],
           'Escenario_Pactos': ['cod_pacto', 'letra_pacto', 'glosa_pacto'],
           'Escenario_SubPactos': ['cod_subp', 'cod_pacto', 'glosa_subp'],
           'Escenario_Partidos': ['cod_part', 'glosa_part', 'sigla_part'],
           'Escenario_Zonas': ['cod_zona', 'glosa_zona', 'tipo_zona', 'orden_zona'],
           'Escenario_ZonasPadre': ['cod_zona', 'tipo_zona', 'cod_zona_pad', 'tipo_zona_pad'],
           'VOTACION_8': ['cod_elec', 'ambito', 'cod_ambito', 'cod_zona', 'tipo_zona', 'votos']}

district_seats = [3, 3, 4, 4, 6, 8, 7, 7, 6, 7, 6, 6, 4, 5, 5, 4, 7, 4, 5, 7, 4, 3, 6, 4, 3, 4, 3, 3]