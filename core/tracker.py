import math
from typing import Any, Dict, List, Optional, Tuple

# A velocidade é expressa em "pixels por frame a 30 FPS" para que o limiar de
# 15 px/f do artigo tenha o mesmo significado em vídeos com outras taxas.
FPS_REFERENCIA = 30.0


class PersonTracker:
    """
    Rastreador isolado por vídeo/sessão para manter o histórico de centroides e velocidades.
    Elimina o uso de variáveis globais e permite reset explícito.

    A velocidade é calculada pela distância entre o centroide atual e a última
    observação do mesmo ID, dividida pelo número de frames entre as duas. Assim,
    uma pessoa que some atrás de um obstáculo e reaparece não gera um "salto"
    de velocidade. IDs não vistos há mais de `max_frames_ausente` são descartados.
    """
    def __init__(
        self,
        fps_video: Optional[float] = None,
        max_frames_ausente: int = 30,
        max_pontos_trajetoria: int = 15
    ):
        self.historico_posicoes: Dict[int, Tuple[int, int, int]] = {}
        self.historico_trajetorias: Dict[int, List[Tuple[int, int]]] = {}
        self.max_pontos_trajetoria = max_pontos_trajetoria
        self.max_frames_ausente = max_frames_ausente
        self.fator_fps = self._calcular_fator_fps(fps_video)
        self.frame_atual = 0

    @staticmethod
    def _calcular_fator_fps(fps_video: Optional[float]) -> float:
        if not fps_video or fps_video <= 0 or math.isnan(fps_video):
            return 1.0
        # Em 60 FPS a mesma queda anda metade dos pixels por frame que em 30 FPS
        return float(fps_video) / FPS_REFERENCIA

    def reset(self):
        """Limpa todo o histórico de rastreamento."""
        self.historico_posicoes.clear()
        self.historico_trajetorias.clear()
        self.frame_atual = 0

    def _descartar_ausentes(self):
        limite = self.frame_atual - self.max_frames_ausente
        ausentes = [obj_id for obj_id, (_, _, f) in self.historico_posicoes.items() if f < limite]
        for obj_id in ausentes:
            del self.historico_posicoes[obj_id]
            self.historico_trajetorias.pop(obj_id, None)

    def atualizar(
        self,
        boxes: List[List[float]],
        track_ids: List[Optional[int]]
    ) -> List[Dict[str, Any]]:
        """
        Recebe as caixas delimitadoras e IDs rastreados do YOLO (None quando o
        rastreador ainda não atribuiu ID). Deve ser chamado uma vez por frame.
        Calcula o centroide, velocidade instantânea e histórico de movimento.
        """
        self.frame_atual += 1
        self._descartar_ausentes()
        resultados = []

        for box, obj_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            largura = max(0.0, float(x2 - x1))
            altura = max(0.0, float(y2 - y1))
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            velocidade = 0.0
            trajetoria = [(cx, cy)]

            # Sem ID não há como saber quem é a pessoa no frame anterior
            if obj_id is not None:
                obj_id = int(obj_id)
                if obj_id in self.historico_posicoes:
                    cx_ant, cy_ant, frame_ant = self.historico_posicoes[obj_id]
                    frames_decorridos = max(1, self.frame_atual - frame_ant)
                    distancia = math.sqrt((cx - cx_ant) ** 2 + (cy - cy_ant) ** 2)
                    velocidade = distancia / frames_decorridos * self.fator_fps

                self.historico_posicoes[obj_id] = (cx, cy, self.frame_atual)

                historico = self.historico_trajetorias.setdefault(obj_id, [])
                historico.append((cx, cy))
                if len(historico) > self.max_pontos_trajetoria:
                    historico.pop(0)
                trajetoria = list(historico)

            resultados.append({
                "id": obj_id,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "largura": largura,
                "altura": altura,
                "centroide": (cx, cy),
                "velocidade": float(velocidade),
                "trajetoria": trajetoria
            })

        return resultados
