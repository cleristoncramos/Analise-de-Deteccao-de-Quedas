"""Análise ao vivo da câmera do navegador (WebRTC), com o mesmo pipeline do Estúdio de Inferência.

O `cv2.VideoCapture(0)` abre a câmera do computador que executa o Streamlit; no
Streamlit Cloud esse computador é um servidor sem câmera. Com o streamlit-webrtc
a imagem vem da câmera de quem acessa a página, e cada quadro é processado no
servidor por um `ProcessadorCamera` exclusivo daquela conexão.
"""
import json
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass

import av
import cv2
import streamlit as st
from streamlit_webrtc import VideoProcessorBase

from core.eventos import AgrupadorEventos, chave_evento
from core.heuristics import analisar_interacoes
from core.ice_servers import montar_ice_servers, tem_turn
from core.tracker import PersonTracker
from core.video_processor import desenhar_anotacoes, redimensionar_frame
from core.yolo_engine import processar_frame_detector, reiniciar_rastreamento

FPS_REFERENCIA = 30.0
TRADUCOES = {
    "start": "Ligar câmera",
    "stop": "Desligar câmera",
    "select_device": "Selecionar câmera",
    "media_api_not_available": "Este navegador não permite acesso à câmera.",
    "device_ask_permission": "Permita o acesso à câmera no navegador.",
    "device_not_available": "Nenhuma câmera encontrada.",
    "device_access_denied": "O acesso à câmera foi negado.",
}


def _para_dict(valor):
    """Converte a seção dos secrets (AttrDict aninhado) em dict/list comuns."""
    if isinstance(valor, Mapping):
        return {k: _para_dict(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_para_dict(v) for v in valor]
    return valor


@st.cache_data(ttl=3600, show_spinner=False)
def _configuracao_em_cache(turn_json: str):
    return montar_ice_servers(json.loads(turn_json) if turn_json else None)


def obter_configuracao_rtc():
    """Retorna (rtc_configuration, com_turn, erro) a partir da seção [turn] dos secrets."""
    try:
        turn = st.secrets.get("turn")
    except Exception:  # sem secrets.toml (execução local)
        turn = None
    turn_json = json.dumps(_para_dict(turn), sort_keys=True) if turn else ""
    servidores, _origem, erro = _configuracao_em_cache(turn_json)
    if erro:
        _configuracao_em_cache.clear()  # não guarda a falha: tenta de novo na próxima execução
    return {"iceServers": servidores}, tem_turn(servidores), erro


@dataclass
class ParametrosAnalise:
    executar_com: bool
    executar_sem: bool
    conf_min: float
    limiar_ar: float
    limiar_vel: float
    limiar_prox: int
    resolucao_alvo: int
    corrigir_fps: bool
    carencia_s: float
    modelo: str


class ProcessadorCamera(VideoProcessorBase):
    """Recebe os quadros da câmera do navegador, aplica a heurística e devolve o quadro anotado.

    Roda em uma thread do streamlit-webrtc, fora do script do Streamlit: por isso os eventos
    vão para uma fila e as métricas ficam atrás de um lock, lidas pela página a cada atualização.
    """

    def __init__(self, modelo, parametros: ParametrosAnalise):
        self.modelo = modelo
        self.parametros = parametros
        self.eventos = queue.Queue()
        self._lock = threading.Lock()
        reiniciar_rastreamento(modelo)
        self.tracker = PersonTracker(fps_video=FPS_REFERENCIA if parametros.corrigir_fps else None)
        carencia = round(FPS_REFERENCIA * parametros.carencia_s)
        self.agrupador_com = AgrupadorEventos(carencia)
        self.agrupador_sem = AgrupadorEventos(carencia)
        self.numero_frame = 0
        self.fps = 0.0
        self.pessoas = 0
        self.quedas_com = 0
        self.quedas_sem = 0
        self.ultima_queda_frame = None
        self._ultimo_instante = None

    def atualizar_parametros(self, parametros: ParametrosAnalise):
        with self._lock:
            self.parametros = parametros

    def metricas(self) -> dict:
        with self._lock:
            return {"frame": self.numero_frame, "fps": self.fps, "pessoas": self.pessoas,
                    "quedas_com": self.quedas_com, "quedas_sem": self.quedas_sem,
                    "ultima_queda_frame": self.ultima_queda_frame}

    def _registrar(self, eventos, agrupador, rotulo_modo) -> int:
        quedas = 0
        for ev in eventos:
            chave = chave_evento(ev)
            if chave is not None and not agrupador.novo_evento(chave, self.numero_frame):
                continue
            if ev["tipo"] == "QUEDA":
                quedas += 1
            self.eventos.put({
                "Frame": self.numero_frame,
                "Timestamp": time.strftime("%H:%M:%S"),
                "Tipo": ev["tipo"],
                "Postura": ev["postura"],
                "Velocidade": ev["velocidade"],
                "Detalhes": ev["detalhes"],
                "Modo": rotulo_modo,
                "Fonte": "Câmera do navegador",
                "Modelo": self.parametros.modelo,
            })
        return quedas

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        with self._lock:
            p = self.parametros
        agora = time.time()
        if self._ultimo_instante is not None:
            intervalo = max(1e-3, agora - self._ultimo_instante)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / intervalo) if self.fps else 1.0 / intervalo
            # A câmera do navegador não tem FPS fixo: a velocidade usa a taxa medida
            self.tracker.fator_fps = (self.fps / FPS_REFERENCIA) if p.corrigir_fps else 1.0
        self._ultimo_instante = agora
        self.numero_frame += 1

        imagem = redimensionar_frame(frame.to_ndarray(format="bgr24"), largura_alvo=p.resolucao_alvo)
        pessoas = processar_frame_detector(self.modelo, imagem, tracker=self.tracker, confianca=p.conf_min)
        alertas_prox = analisar_interacoes([q["centroide"] for q in pessoas], distancia_minima=p.limiar_prox)

        quadros = []
        queda = False
        if p.executar_sem:
            anotado, eventos = desenhar_anotacoes(imagem, pessoas, alertas_prox, modo_velocidade=False,
                                                  limiar_queda=p.limiar_vel, limiar_ar=p.limiar_ar)
            novas = self._registrar(eventos, self.agrupador_sem, "Sem Velocidade")
            with self._lock:
                self.quedas_sem += novas
            quadros.append(anotado)
            queda = queda or (not p.executar_com and any(e["tipo"] == "QUEDA" for e in eventos))
        if p.executar_com:
            anotado, eventos = desenhar_anotacoes(imagem, pessoas, alertas_prox, modo_velocidade=True,
                                                  limiar_queda=p.limiar_vel, limiar_ar=p.limiar_ar)
            novas = self._registrar(eventos, self.agrupador_com, "Com Velocidade")
            with self._lock:
                self.quedas_com += novas
            quadros.append(anotado)
            queda = queda or any(e["tipo"] == "QUEDA" for e in eventos)

        with self._lock:
            self.pessoas = len(pessoas)
            if queda:
                self.ultima_queda_frame = self.numero_frame

        saida = cv2.hconcat(quadros) if len(quadros) > 1 else quadros[0]
        return av.VideoFrame.from_ndarray(saida, format="bgr24")
