import math

import pytest

from core.heuristics import (
    analisar_interacoes,
    analisar_postura,
    calcular_velocidade_normalizada,
    classificar_acao,
    confirmar_queda,
)


@pytest.mark.parametrize(
    ("largura", "altura", "esperado"),
    [
        (200, 100, "Deitado"),           # AR 0.5
        (100, 75, "Sentado/Agachado"),   # AR 0.75 (limite inclusivo)
        (100, 120, "Sentado/Agachado"),  # AR 1.2 (limite inclusivo)
        (100, 250, "Em pe"),             # AR 2.5
        (0, 100, "Indefinido"),
    ],
)
def test_analisar_postura(largura, altura, esperado):
    assert analisar_postura(largura, altura)[0] == esperado


def test_classificar_acao():
    assert classificar_acao(0.5) == "Mexendo-se Pouco"
    assert classificar_acao(10) == "Andando"
    assert classificar_acao(30) == "Correndo"


def test_confirmar_queda_bimodal_exige_velocidade():
    assert confirmar_queda("Deitado", 20.0, limiar_queda=15.0)
    assert not confirmar_queda("Deitado", 5.0, limiar_queda=15.0)
    assert not confirmar_queda("Em pe", 50.0, limiar_queda=15.0)


def test_confirmar_queda_estatico_ignora_velocidade():
    assert confirmar_queda("Deitado", 0.0, usar_velocidade=False)
    assert not confirmar_queda("Sentado/Agachado", 0.0, usar_velocidade=False)


def test_analisar_interacoes():
    alertas = analisar_interacoes([(0, 0), (30, 40), (500, 500)], distancia_minima=60)
    assert len(alertas) == 1
    assert (alertas[0]["idx_a"], alertas[0]["idx_b"]) == (0, 1)
    assert alertas[0]["distancia"] == pytest.approx(50.0)


def test_velocidade_normalizada_independe_da_escala():
    # A mesma queda vista de perto e de longe: deslocamento e caixa escalam juntos
    perto = calcular_velocidade_normalizada(40.0, 200, 450)
    longe = calcular_velocidade_normalizada(10.0, 50, 112.5)
    assert perto == pytest.approx(longe)
    assert perto == pytest.approx(40.0 / math.sqrt(200 * 450) * 100)
    assert calcular_velocidade_normalizada(10.0, 0, 100) == 0.0
