"""Confere se as métricas publicadas batem com as matrizes de confusão do artigo."""
import json
from pathlib import Path

import pytest

DADOS = json.loads((Path(__file__).parent.parent / "data" / "tcc_benchmark_data.json").read_text(encoding="utf-8"))
TABELAS = [k for k in DADOS if k.startswith("tabela_")]
LINHAS = [(t, linha) for t in TABELAS for linha in DADOS[t]]


def _metricas(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    precisao = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precisao * recall / (precisao + recall) if precisao + recall else 0.0
    return {"acuracia": (tn + tp) / total, "precisao": precisao, "recall": recall, "f1_score": f1}


@pytest.mark.parametrize(("tabela", "linha"), LINHAS, ids=[f"{t}-{linha['cenario']}-{linha['modelo']}" for t, linha in LINHAS])
def test_metricas_batem_com_matriz(tabela, linha):
    calculado = _metricas(linha["tn"], linha["fp"], linha["fn"], linha["tp"])
    for nome, valor in calculado.items():
        assert linha[nome] == pytest.approx(valor, abs=0.006), nome


def test_tabela4_e_soma_das_tabelas_1_a_3():
    def por_modelo(tabela):
        return {(r["cenario"], r["modelo"]): r for r in DADOS[tabela]}

    t1, t2, t4 = por_modelo("tabela_1_simulado"), por_modelo("tabela_2_ur_fall"), por_modelo("tabela_4_geral")
    publico = {r["cenario"]: r for r in DADOS["tabela_3_publico"]}
    for chave, geral in t4.items():
        for campo in ("tn", "fp", "fn", "tp"):
            soma = t1[chave][campo] + t2[chave][campo] + publico[chave[0]][campo]
            assert geral[campo] == soma, f"{chave} {campo}"


def test_todos_os_modelos_com_velocidade_tem_38_quedas():
    for linha in DADOS["tabela_4_geral"]:
        assert linha["tp"] + linha["fn"] == 38, linha["modelo"]
