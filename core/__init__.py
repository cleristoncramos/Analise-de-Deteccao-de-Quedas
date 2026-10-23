"""
Módulo Core para o Sistema de Detecção e Análise de Quedas (TCC - UFPI).

O motor YOLO (`core.yolo_engine`) e o desenho (`core.video_processor`) não são
importados aqui para que as heurísticas possam ser usadas e testadas sem
Ultralytics/PyTorch/OpenCV instalados.
"""
from .eventos import AgrupadorEventos, chave_evento
from .heuristics import (
    analisar_interacoes,
    analisar_postura,
    calcular_velocidade_normalizada,
    classificar_acao,
    confirmar_queda,
)
from .tracker import PersonTracker

__all__ = [
    "AgrupadorEventos",
    "PersonTracker",
    "analisar_interacoes",
    "analisar_postura",
    "calcular_velocidade_normalizada",
    "chave_evento",
    "classificar_acao",
    "confirmar_queda",
]
