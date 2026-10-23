from typing import Dict, Hashable, Optional


class AgrupadorEventos:
    """
    Agrupa detecções quadro a quadro em eventos.

    Uma mesma queda costuma satisfazer a heurística em vários frames seguidos,
    e uma dupla próxima fica próxima por segundos. Sem agrupamento, cada frame
    vira uma linha no log. Aqui uma chave (ex.: ("QUEDA", id)) só gera um novo
    evento depois de passar `frames_carencia` frames sem ser vista.
    """

    def __init__(self, frames_carencia: int = 60):
        self.frames_carencia = max(0, int(frames_carencia))
        self.ultimo_frame: Dict[Hashable, int] = {}

    @classmethod
    def a_partir_do_fps(cls, fps_video: Optional[float], segundos: float = 2.0) -> "AgrupadorEventos":
        fps = fps_video if fps_video and fps_video > 0 else 30.0
        return cls(frames_carencia=round(fps * segundos))

    def reset(self):
        self.ultimo_frame.clear()

    def novo_evento(self, chave: Hashable, numero_frame: int) -> bool:
        """Registra que `chave` ocorreu no frame e diz se isso inicia um evento novo."""
        anterior = self.ultimo_frame.get(chave)
        self.ultimo_frame[chave] = numero_frame
        return anterior is None or numero_frame - anterior > self.frames_carencia


def chave_evento(evento: Dict) -> Optional[Hashable]:
    """
    Chave de agrupamento de um evento produzido por `desenhar_anotacoes`.
    Retorna None quando não há ID estável (cada ocorrência conta como evento).
    """
    if evento["tipo"] == "QUEDA":
        return None if evento.get("id") is None else ("QUEDA", evento["id"])
    if evento["tipo"] == "PROXIMIDADE":
        ids = evento.get("ids")
        if not ids or None in ids:
            return None
        return ("PROXIMIDADE", tuple(sorted(ids)))
    return None
