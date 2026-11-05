"""Métricas da avaliação do conjunto do UR Fall e resultados pré-calculados dos 5 modelos."""
from core.catalogo_videos import carregar_catalogo
from core.metricas_conjunto import (
    carregar_pre_calculados,
    classificar,
    linha_do_artigo,
    metricas,
    parametros_compativeis,
    selecionar_videos,
)


def _matriz(m):
    return m["tn"], m["fp"], m["fn"], m["tp"]


def test_classificacao_e_metricas_por_evento():
    assert [classificar("queda", 2), classificar("queda", 0), classificar("nao_queda", 1),
            classificar("nao_queda", 0)] == ["VP", "FN", "FP", "VN"]
    m = metricas(["VP"] * 18 + ["FN"] * 12 + ["VN"] * 30)
    assert _matriz(m) == (30, 0, 12, 18) and m["total"] == 60
    assert round(m["acuracia"], 3) == 0.8 and m["precisao"] == 1.0 and m["recall"] == 0.6
    assert round(m["f1_score"], 3) == 0.75


def test_metricas_sem_videos_nao_dividem_por_zero():
    m = metricas([])
    assert m["total"] == 0 and m["acuracia"] == m["precisao"] == m["recall"] == m["f1_score"] == 0.0


def test_selecao_de_videos_respeita_o_conjunto_do_artigo():
    videos = carregar_catalogo()["videos"]
    assert len(selecionar_videos(videos, n_adl=30)) == 60
    assert len(selecionar_videos(videos, n_adl=40)) == 70
    assert {v["rotulo"] for v in selecionar_videos(videos, somente_quedas=True)} == {"queda"}
    assert [v["id"] for v in selecionar_videos(videos, limite=2)] == ["fall-01", "fall-02"]


def test_pre_calculados_cobrem_os_70_videos_dos_5_modelos():
    pre = carregar_pre_calculados()
    assert sorted(pre) == ["yolo11l", "yolo11m", "yolo11n", "yolo11s", "yolo11x"]
    ids = {v["id"] for v in carregar_catalogo()["videos"]}
    assert all(set(videos) == ids for videos in pre.values())


def test_pre_calculado_do_v11m_reproduz_a_tabela_2_com_velocidade():
    # O v11m com velocidade e 30 ADL bateu exatamente com o artigo na avaliação pelo CLI
    pre = carregar_pre_calculados()["yolo11m"]
    videos = selecionar_videos(carregar_catalogo()["videos"], n_adl=30)
    m = metricas(classificar(v["rotulo"], pre[v["id"]]["com"]) for v in videos)
    artigo = linha_do_artigo("yolo11m.pt", "com")
    assert _matriz(m) == (artigo["tn"], artigo["fp"], artigo["fn"], artigo["tp"]) == (30, 0, 13, 17)


def test_linha_do_artigo_e_compatibilidade_dos_parametros():
    assert linha_do_artigo("yolo11s.pt", "com")["tp"] == 18
    assert linha_do_artigo("yolo11s.pt", "sem") is None  # a Tabela 2 só tem o cenário sem velocidade para o v11m
    padrao = {"conf": 0.5, "ar": 0.75, "vel": 15.0, "largura": 1080, "corrigir_fps": True, "pular": 1}
    assert parametros_compativeis(**padrao)
    assert not parametros_compativeis(**{**padrao, "vel": 20.0})
    assert not parametros_compativeis(**{**padrao, "pular": 2})
