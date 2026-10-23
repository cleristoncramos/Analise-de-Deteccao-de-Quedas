from core.eventos import AgrupadorEventos, chave_evento


def test_mesma_queda_em_frames_seguidos_gera_um_evento():
    ag = AgrupadorEventos(frames_carencia=10)
    novos = [ag.novo_evento(("QUEDA", 1), f) for f in range(1, 8)]
    assert novos == [True] + [False] * 6


def test_nova_queda_apos_carencia():
    ag = AgrupadorEventos(frames_carencia=10)
    assert ag.novo_evento(("QUEDA", 1), 1)
    assert not ag.novo_evento(("QUEDA", 1), 11)
    assert ag.novo_evento(("QUEDA", 1), 30)


def test_pessoas_diferentes_sao_eventos_diferentes():
    ag = AgrupadorEventos(frames_carencia=10)
    assert ag.novo_evento(("QUEDA", 1), 1)
    assert ag.novo_evento(("QUEDA", 2), 1)


def test_carencia_a_partir_do_fps():
    assert AgrupadorEventos.a_partir_do_fps(25, 2.0).frames_carencia == 50
    assert AgrupadorEventos.a_partir_do_fps(0, 2.0).frames_carencia == 60


def test_chave_evento():
    assert chave_evento({"tipo": "QUEDA", "id": 4}) == ("QUEDA", 4)
    assert chave_evento({"tipo": "QUEDA", "id": None}) is None
    assert chave_evento({"tipo": "PROXIMIDADE", "ids": (5, 2)}) == ("PROXIMIDADE", (2, 5))
    assert chave_evento({"tipo": "PROXIMIDADE", "ids": (5, None)}) is None
