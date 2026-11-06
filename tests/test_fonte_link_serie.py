"""Fonte por link (validação e YouTube simulado) e série temporal do gráfico de movimento."""
import pytest

from core.fonte_link import ErroLink, baixar_youtube, eh_youtube, validar_link
from core.serie_movimento import SerieMovimento


@pytest.mark.parametrize("url", ["https://exemplo.org/v.mp4", "rtsp://192.168.0.10:554/stream",
                                 "http://camera.local/live.m3u8", "rtmp://servidor/app/chave"])
def test_links_aceitos(url):
    assert validar_link(f"  {url}  ") == url


@pytest.mark.parametrize("url", ["", "file:///etc/passwd", "C:/videos/a.mp4", "ftp://x/y.mp4", "https://"])
def test_links_recusados(url):
    with pytest.raises(ErroLink):
        validar_link(url)


def test_reconhece_youtube():
    assert eh_youtube("https://www.youtube.com/watch?v=abc")
    assert eh_youtube("https://youtu.be/abc")
    assert eh_youtube("https://m.youtube.com/watch?v=abc")
    assert not eh_youtube("https://naoyoutube.com/watch?v=abc")
    assert not eh_youtube("https://exemplo.org/youtube.com.mp4")


def test_youtube_bloqueado_vira_mensagem_clara(tmp_path):
    def baixar(url, pasta, limite):
        raise RuntimeError("ERROR: Sign in to confirm you're not a bot")

    with pytest.raises(ErroLink, match="Upload de Arquivo"):
        baixar_youtube("https://youtu.be/abc", tmp_path, baixar=baixar)


def test_youtube_baixado_retorna_caminho(tmp_path):
    def baixar(url, pasta, limite):
        assert limite == 60
        destino = pasta / "youtube.mp4"
        destino.write_bytes(b"video")
        return destino

    assert baixar_youtube("https://youtu.be/abc", tmp_path, 60, baixar=baixar).read_bytes() == b"video"


def _pessoa(id_, largura, altura, v):
    return {"id": id_, "largura": largura, "altura": altura, "velocidade": v}


def test_serie_segue_a_pessoa_com_queda():
    serie = SerieMovimento()
    for quadro in range(1, 11):
        pessoas = [_pessoa(1, 100, 200, 2.0)]
        if quadro > 6:
            pessoas.append(_pessoa(2, 200, 100, 20.0))
        serie.registrar(quadro, pessoas + [_pessoa(None, 50, 50, 0.0)])
    assert serie.pessoa_principal() == 1  # a mais observada
    serie.registrar_queda(9, 2)
    dados = serie.dados_principal()
    assert dados["id"] == 2 and dados["frame"] == [7, 8, 9, 10]
    assert dados["ar"][0] == 0.5 and dados["v"][0] == 20.0 and dados["quedas"] == [9]


def test_grafico_de_movimento():
    pytest.importorskip("plotly")
    from components.chart_builder import criar_grafico_movimento

    fig = criar_grafico_movimento({"frame": [1, 2, 3], "ar": [2.0, 1.0, 0.5], "v": [1.0, 18.0, 3.0],
                                   "quedas": [2]}, limiar_ar=0.75, limiar_vel=15.0)
    assert len(fig.data) == 2  # razão de aspecto e velocidade, em painéis alinhados
    tracejadas = [f for f in fig.layout.shapes if f.type == "line" and f.line.dash == "dot"]
    assert sorted(f.y0 for f in tracejadas) == [0.75, 15.0]  # os dois limiares
    quedas = [f for f in fig.layout.shapes if f.type == "rect" and f.x0 is not None and f.x0 < 2 < f.x1]
    assert len(quedas) == 2  # a queda do quadro 2 marcada nos dois painéis
    assert fig.layout.yaxis2.title.text == "v (px/quadro)"
