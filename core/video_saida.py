"""Gravação do vídeo anotado da análise em MP4 H.264, reproduzível no navegador.

A análise roda mais devagar que o vídeo (cada quadro passa pelo YOLO), então mostrar
os quadros ao vivo dá a impressão de câmera lenta. O resultado é gravado em um MP4 com
a taxa de quadros original, para ser assistido em velocidade real ao final.

Usa o PyAV (já instalado com o streamlit-webrtc), cujos pacotes trazem o codificador
libx264. Sem ele, a gravação é desativada e a análise segue normalmente.
"""
from pathlib import Path
from typing import Optional

import numpy as np


class GravadorVideo:
    def __init__(self, caminho: Path, fps: float, largura_max: int = 720, crf: int = 28):
        import av

        self.caminho = Path(caminho)
        self.fps = fps if fps and fps > 0 else 30.0
        self.largura_max = largura_max
        self.crf = crf
        self._av = av
        self._saida = av.open(str(self.caminho), mode="w", options={"movflags": "+faststart"})
        self._fluxo = None
        self.quadros = 0

    def _abrir_fluxo(self, largura: int, altura: int):
        from fractions import Fraction

        fluxo = self._saida.add_stream("libx264", rate=Fraction(round(self.fps * 1000), 1000))
        fluxo.width, fluxo.height = largura, altura
        fluxo.pix_fmt = "yuv420p"
        fluxo.options = {"crf": str(self.crf), "preset": "veryfast"}
        self._fluxo = fluxo

    def escrever(self, quadro_bgr: np.ndarray):
        import cv2

        altura, largura = quadro_bgr.shape[:2]
        if largura > self.largura_max:
            altura = round(altura * self.largura_max / largura)
            largura = self.largura_max
        largura -= largura % 2  # yuv420p exige dimensões pares
        altura -= altura % 2
        if self._fluxo is None:
            self._abrir_fluxo(largura, altura)
        quadro = cv2.resize(quadro_bgr, (self._fluxo.width, self._fluxo.height))
        for pacote in self._fluxo.encode(self._av.VideoFrame.from_ndarray(quadro, format="bgr24")):
            self._saida.mux(pacote)
        self.quadros += 1

    def fechar(self) -> Optional[Path]:
        if self._fluxo is not None:
            for pacote in self._fluxo.encode():
                self._saida.mux(pacote)
        self._saida.close()
        return self.caminho if self.quadros else None


def criar_gravador(caminho: Path, fps: float, largura_max: int = 720) -> Optional[GravadorVideo]:
    try:
        return GravadorVideo(caminho, fps, largura_max)
    except Exception:
        return None
