"""Contagem de eventos e comparação com a Tabela 2 do script de avaliação do UR Fall."""
import importlib.util
from pathlib import Path

ARQUIVO = Path(__file__).resolve().parent.parent / "tools" / "avaliar_urfall.py"
especificacao = importlib.util.spec_from_file_location("avaliar_urfall", ARQUIVO)
avaliar_urfall = importlib.util.module_from_spec(especificacao)
especificacao.loader.exec_module(avaliar_urfall)


def test_lista_tem_30_quedas_e_40_adl_da_camera_0():
    videos = avaliar_urfall.lista_videos()
    assert sum(v["rotulo"] == "queda" for v in videos) == 30
    assert sum(v["rotulo"] == "nao_queda" for v in videos) == 40
    assert videos[0]["url"].endswith("/fall-01-cam0.mp4") and videos[-1]["arquivo"] == "adl-40-cam0.mp4"


def _linhas(quedas_fall, quedas_adl):
    linhas = []
    for i, q in enumerate(quedas_fall, start=1):
        linhas.append({"modelo": "yolo11s", "modo": "com", "rotulo": "queda", "indice": i, "quedas": q,
                       "classe": avaliar_urfall.classificar("queda", q)})
    for i, q in enumerate(quedas_adl, start=1):
        linhas.append({"modelo": "yolo11s", "modo": "com", "rotulo": "nao_queda", "indice": i, "quedas": q,
                       "classe": avaliar_urfall.classificar("nao_queda", q)})
    return linhas


def test_resumo_com_30_e_40_adl():
    # 18 de 30 quedas detectadas; alarme falso só na ADL 35 (fora das 30 primeiras)
    linhas = _linhas([1] * 18 + [0] * 12, [0] * 34 + [2] + [0] * 5)
    m30 = avaliar_urfall.resumir(linhas, 30)[("yolo11s", "com")]
    m40 = avaliar_urfall.resumir(linhas, None)[("yolo11s", "com")]
    assert (m30["tn"], m30["fp"], m30["fn"], m30["tp"]) == (30, 0, 12, 18)
    assert round(m30["acuracia"], 3) == 0.8 and m30["precisao"] == 1.0
    assert (m40["tn"], m40["fp"]) == (39, 1)


def test_comparacao_com_a_tabela_2():
    linhas = _linhas([1] * 18 + [0] * 12, [0] * 40)
    texto = avaliar_urfall.comparacao_markdown(avaliar_urfall.resumir(linhas, 30), avaliar_urfall.resumir(linhas, None),
                                               avaliar_urfall.tabela_artigo(), "# teste")
    # YOLOv11s com velocidade no artigo: TN 30, FP 0, FN 12, TP 18, ou seja, diferença zero em todas as células
    assert "| YOLOv11s | Com Velocidade | 30/0/12/18 | 30/0/12/18 | +0/+0/+0/+0 |" in texto
