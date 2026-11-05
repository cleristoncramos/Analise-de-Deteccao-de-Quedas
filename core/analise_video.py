"""Laço de análise de um vídeo inteiro, compartilhado pelo CLI e pela avaliação do conjunto no app."""
from typing import Callable, Dict, List, Optional, Sequence

import cv2

from core.catalogo_videos import recortar_quadro
from core.eventos import AgrupadorEventos, chave_evento
from core.heuristics import analisar_interacoes
from core.horario import data_hora_local
from core.tracker import PersonTracker
from core.video_processor import desenhar_anotacoes, redimensionar_frame
from core.yolo_engine import processar_frame_detector, reiniciar_rastreamento


def analisar_video(cap, modelo, modos: Sequence[str] = ("com",), conf: float = 0.5, limiar_ar: float = 0.75,
                   limiar_vel: float = 15.0, dist_prox: float = 60.0, largura: int = 1080,
                   corrigir_fps: bool = True, carencia_s: float = 2.0, por_frame: bool = False,
                   recorte: Optional[str] = None, janela: bool = False,
                   ao_quadro: Optional[Callable[[int], None]] = None) -> Dict:
    """Processa o vídeo aberto em `cap` e devolve os eventos registrados por modo ('com' e/ou 'sem').

    Uma única inferência por quadro alimenta todos os modos pedidos, como no Estúdio de Inferência:
    rodar os dois modos juntos dá o mesmo resultado que rodá-los separadamente, com metade do custo.
    `ao_quadro(numero_do_quadro)` é chamado a cada quadro (barra de progresso, pausa cooperativa).
    """
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    reiniciar_rastreamento(modelo)
    tracker = PersonTracker(fps_video=fps_video if corrigir_fps else None)
    agrupadores = {modo: AgrupadorEventos.a_partir_do_fps(fps_video, carencia_s) for modo in modos}
    eventos: Dict[str, List[list]] = {modo: [] for modo in modos}
    quedas = {modo: 0 for modo in modos}
    recorte = None if recorte in (None, "nenhum") else recorte
    numero_frame = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            numero_frame += 1

            frame = redimensionar_frame(recortar_quadro(frame, recorte), largura_alvo=largura)
            pessoas = processar_frame_detector(modelo, frame, tracker=tracker, confianca=conf)
            interacoes = analisar_interacoes([p["centroide"] for p in pessoas], distancia_minima=dist_prox)
            agora = data_hora_local()
            frame_anotado = frame
            for modo in modos:
                frame_anotado, evs = desenhar_anotacoes(
                    frame, pessoas, interacoes,
                    modo_velocidade=modo == "com", limiar_queda=limiar_vel, limiar_ar=limiar_ar
                )
                for ev in evs:
                    chave = chave_evento(ev)
                    if not por_frame and chave is not None and not agrupadores[modo].novo_evento(chave, numero_frame):
                        continue
                    if ev["tipo"] == "QUEDA":
                        quedas[modo] += 1
                    eventos[modo].append([numero_frame, agora, ev["tipo"], ev["detalhes"], ev["postura"],
                                          ev["velocidade"]])

            if ao_quadro is not None:
                ao_quadro(numero_frame)
            if janela:
                cv2.putText(frame_anotado, f"Pessoas: {len(pessoas)}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow("Sistema de Seguranca", frame_anotado)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if janela:
            cv2.destroyAllWindows()
    return {"frames": numero_frame, "fps": fps_video, "eventos": eventos, "quedas": quedas}
