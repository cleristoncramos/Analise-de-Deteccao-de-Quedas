"""Catálogo do UR Fall e cache de vídeos, com download simulado (sem rede)."""
import json

import numpy as np
import pytest

from core.catalogo_videos import (
    ErroDownloadVideo,
    avaliar_resultado,
    carregar_catalogo,
    obter_video,
    recortar_quadro,
)

MP4_FALSO = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 100
VIDEO = {"id": "fall-01", "nome": "Queda 01", "arquivo": "fall-01-cam0.mp4",
         "url": "https://exemplo.org/fall-01-cam0.mp4", "rotulo": "queda", "recorte": "direita"}


def test_catalogo_do_repositorio_e_valido_e_equilibrado():
    catalogo = carregar_catalogo()
    rotulos = [v["rotulo"] for v in catalogo["videos"]]
    assert 6 <= len(rotulos) <= 10
    assert rotulos.count("queda") == rotulos.count("nao_queda")
    assert len({v["id"] for v in catalogo["videos"]}) == len(rotulos)
    assert all(v["url"].startswith("https://fenix.ur.edu.pl/mkepski/ds/data/") for v in catalogo["videos"])
    assert "Kwolek" in catalogo["citacao"].title()


def test_catalogo_com_campo_faltando_e_rejeitado(tmp_path):
    caminho = tmp_path / "catalogo.json"
    caminho.write_text(json.dumps({"videos": [{"id": "x", "nome": "x", "arquivo": "x.mp4", "rotulo": "queda"}]}))
    with pytest.raises(ValueError, match="url"):
        carregar_catalogo(caminho)


def test_baixa_uma_vez_e_reaproveita_o_cache(tmp_path):
    chamadas = []

    def baixar(url, destino, tempo_limite):
        chamadas.append(url)
        destino.write(MP4_FALSO)

    primeiro = obter_video(VIDEO, tmp_path, baixar)
    segundo = obter_video(VIDEO, tmp_path, baixar)
    assert primeiro == segundo == tmp_path / "fall-01-cam0.mp4"
    assert primeiro.read_bytes() == MP4_FALSO
    assert chamadas == [VIDEO["url"]]


def test_falha_de_rede_nao_deixa_arquivo_no_cache(tmp_path):
    def baixar(url, destino, tempo_limite):
        destino.write(b"pedaco")
        raise OSError("sem rede")

    with pytest.raises(ErroDownloadVideo, match="sem rede"):
        obter_video(VIDEO, tmp_path, baixar)
    assert list(tmp_path.iterdir()) == []


def test_resposta_que_nao_e_video_e_rejeitada(tmp_path):
    def baixar(url, destino, tempo_limite):
        destino.write(b"<html>pagina de erro</html>")

    with pytest.raises(ErroDownloadVideo, match="não é um vídeo"):
        obter_video(VIDEO, tmp_path, baixar)
    assert list(tmp_path.iterdir()) == []


def test_recorte_mantem_metade_rgb():
    quadro = np.zeros((240, 640, 3), dtype=np.uint8)
    quadro[:, 320:] = 255
    rgb = recortar_quadro(quadro, "direita")
    assert rgb.shape == (240, 320, 3) and rgb.min() == 255
    assert recortar_quadro(quadro, None) is quadro


@pytest.mark.parametrize(("rotulo", "houve_queda", "esperado"), [
    ("queda", True, "VP"), ("queda", False, "FN"), ("nao_queda", True, "FP"), ("nao_queda", False, "VN"),
])
def test_avaliar_resultado(rotulo, houve_queda, esperado):
    assert avaliar_resultado(rotulo, houve_queda) == esperado
