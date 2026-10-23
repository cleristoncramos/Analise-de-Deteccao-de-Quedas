import pytest

from core.tracker import PersonTracker


def caixa(cx, cy, w=50, h=100):
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_velocidade_entre_frames_consecutivos():
    t = PersonTracker()
    assert t.atualizar([caixa(100, 100)], [1])[0]["velocidade"] == 0.0
    assert t.atualizar([caixa(103, 104)], [1])[0]["velocidade"] == pytest.approx(5.0)


def test_oclusao_nao_gera_salto_de_velocidade():
    t = PersonTracker()
    t.atualizar([caixa(100, 100)], [1])
    for _ in range(9):  # pessoa oculta por 9 frames
        t.atualizar([], [])
    r = t.atualizar([caixa(200, 100)], [1])[0]
    # 100 px em 10 frames = 10 px/f, não 100 px/f
    assert r["velocidade"] == pytest.approx(10.0)


def test_id_ausente_por_muito_tempo_e_descartado():
    t = PersonTracker(max_frames_ausente=5)
    t.atualizar([caixa(100, 100)], [1])
    for _ in range(10):
        t.atualizar([], [])
    assert 1 not in t.historico_posicoes
    assert t.atualizar([caixa(400, 100)], [1])[0]["velocidade"] == 0.0


def test_deteccao_sem_id_nao_herda_velocidade_de_outra_pessoa():
    t = PersonTracker()
    t.atualizar([caixa(100, 100)], [1])
    r = t.atualizar([caixa(100, 100), caixa(500, 500)], [1, None])
    assert r[1]["id"] is None
    assert r[1]["velocidade"] == 0.0
    assert None not in t.historico_posicoes


def test_correcao_por_fps():
    # Em 60 FPS a pessoa anda metade dos pixels por frame; convertido para 30 FPS deve dobrar
    t = PersonTracker(fps_video=60)
    t.atualizar([caixa(100, 100)], [1])
    assert t.atualizar([caixa(105, 100)], [1])[0]["velocidade"] == pytest.approx(10.0)


def test_fps_invalido_nao_altera_velocidade():
    for fps in (None, 0, float("nan")):
        t = PersonTracker(fps_video=fps)
        t.atualizar([caixa(100, 100)], [1])
        assert t.atualizar([caixa(105, 100)], [1])[0]["velocidade"] == pytest.approx(5.0)


def test_trajetoria_limitada():
    t = PersonTracker(max_pontos_trajetoria=3)
    for x in range(10):
        r = t.atualizar([caixa(100 + x, 100)], [7])[0]
    assert len(r["trajetoria"]) == 3


def test_reset():
    t = PersonTracker()
    t.atualizar([caixa(100, 100)], [1])
    t.reset()
    assert t.atualizar([caixa(300, 100)], [1])[0]["velocidade"] == 0.0
