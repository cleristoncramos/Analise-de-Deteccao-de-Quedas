"""Cache de resultados das amostras e gravação do vídeo anotado."""
import numpy as np
import pytest

from core.cache_resultados import carregar_resultado, chave_resultado, salvar_resultado


def test_chave_muda_com_qualquer_parametro():
    base = dict(video="fall-01", modelo="v11s", vel=15.0)
    assert chave_resultado(**base) == chave_resultado(**dict(reversed(list(base.items()))))
    assert chave_resultado(**base) != chave_resultado(**{**base, "vel": 14.0})


def test_salva_e_reaproveita_resultado(tmp_path):
    video = tmp_path / "origem.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    dados = {"numero_frame": 160, "quedas_com": 1, "eventos": [{"Frame": 123, "Tipo": "QUEDA"}]}
    salvar_resultado("abc", dados, video, pasta=tmp_path / "cache")
    lido, video_lido = carregar_resultado("abc", pasta=tmp_path / "cache")
    assert lido == dados
    assert video_lido.read_bytes() == video.read_bytes()
    assert carregar_resultado("outra", pasta=tmp_path / "cache") is None


def test_gravador_gera_mp4_h264(tmp_path):
    pytest.importorskip("av")
    cv2 = pytest.importorskip("cv2")
    from core.video_saida import criar_gravador

    gravador = criar_gravador(tmp_path / "saida.mp4", fps=30, largura_max=720)
    for i in range(15):
        gravador.escrever(np.full((811, 1081, 3), i * 10, dtype=np.uint8))  # dimensões ímpares de propósito
    caminho = gravador.fechar()
    captura = cv2.VideoCapture(str(caminho))
    assert int(captura.get(cv2.CAP_PROP_FRAME_COUNT)) == 15
    assert captura.get(cv2.CAP_PROP_FPS) == pytest.approx(30, abs=0.5)
    assert int(captura.get(cv2.CAP_PROP_FRAME_WIDTH)) % 2 == 0
