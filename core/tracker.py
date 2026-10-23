import math
from typing import Dict, Tuple, List, Any

class PersonTracker:
    """
    Rastreador isolado por vídeo/sessão para manter o histórico de centroides e velocidades.
    Elimina o uso de variáveis globais e permite reset explícito.
    """
    def __init__(self):
        self.historico_posicoes: Dict[int, Tuple[int, int]] = {}
        self.historico_trajetorias: Dict[int, List[Tuple[int, int]]] = {}
        self.max_pontos_trajetoria = 15

    def reset(self):
        """Limpa todo o histórico de rastreamento."""
        self.historico_posicoes.clear()
        self.historico_trajetorias.clear()

    def atualizar(
        self,
        boxes: List[List[float]],
        track_ids: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Recebe as caixas delimitadoras e IDs rastreados do YOLO.
        Calcula o centroide, velocidade instantânea e histórico de movimento.
        """
        resultados = []

        for box, obj_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            largura = max(0.0, float(x2 - x1))
            altura = max(0.0, float(y2 - y1))
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            velocidade = 0.0
            if obj_id in self.historico_posicoes:
                cx_ant, cy_ant = self.historico_posicoes[obj_id]
                velocidade = math.sqrt((cx - cx_ant) ** 2 + (cy - cy_ant) ** 2)

            # Atualiza posição anterior
            self.historico_posicoes[obj_id] = (cx, cy)

            # Atualiza rastro de trajetória
            if obj_id not in self.historico_trajetorias:
                self.historico_trajetorias[obj_id] = []
            self.historico_trajetorias[obj_id].append((cx, cy))
            if len(self.historico_trajetorias[obj_id]) > self.max_pontos_trajetoria:
                self.historico_trajetorias[obj_id].pop(0)

            resultados.append({
                "id": int(obj_id),
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "largura": largura,
                "altura": altura,
                "centroide": (cx, cy),
                "velocidade": float(velocidade),
                "trajetoria": list(self.historico_trajetorias[obj_id])
            })

        return resultados
