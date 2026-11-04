"""Fuso horário local e detecção do modo leve do Streamlit Cloud."""
from datetime import timedelta

from core import ambiente
from core.horario import FUSO_LOCAL, agora_local, data_hora_local, hora_local


def test_horario_em_utc_menos_3():
    assert agora_local().utcoffset() == timedelta(hours=-3)
    assert FUSO_LOCAL is not None
    assert len(hora_local()) == 8 and hora_local().count(":") == 2
    assert len(data_hora_local()) == 19


def test_modo_leve_pode_ser_forcado(monkeypatch):
    monkeypatch.setenv("DQ_MODO_LEVE", "1")
    padroes = ambiente.padroes_execucao()
    assert padroes.modo_leve and padroes.pular_quadros == 1 and padroes.max_quadros > 0 and padroes.threads_torch > 0


def test_fora_do_cloud_segue_a_metodologia(monkeypatch):
    monkeypatch.setenv("DQ_MODO_LEVE", "0")
    padroes = ambiente.padroes_execucao()
    assert not padroes.modo_leve and padroes.pular_quadros == 1 and padroes.max_quadros == 0
