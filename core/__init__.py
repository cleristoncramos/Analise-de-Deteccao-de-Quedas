"""
Módulo Core para o Sistema de Detecção e Análise de Quedas (TCC - UFPI).
"""
from .heuristics import (
    analisar_postura,
    classificar_acao,
    confirmar_queda,
    analisar_interacoes,
    calcular_velocidade_normalizada
)
from .tracker import PersonTracker
from .yolo_engine import carregar_modelo, processar_frame_detector

__all__ = [
    "analisar_postura",
    "classificar_acao",
    "confirmar_queda",
    "analisar_interacoes",
    "calcular_velocidade_normalizada",
    "PersonTracker",
    "carregar_modelo",
    "processar_frame_detector",
]
