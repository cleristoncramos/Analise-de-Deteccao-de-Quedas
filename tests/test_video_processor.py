import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("cv2")

from core.heuristics import analisar_interacoes  # noqa: E402
from core.tracker import PersonTracker  # noqa: E402
from core.video_processor import desenhar_anotacoes, redimensionar_frame  # noqa: E402


def test_redimensionar_mantem_proporcao():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    assert redimensionar_frame(frame, 1080).shape == (607, 1080, 3)


def _cena():
    t = PersonTracker()
    t.atualizar([[100, 100, 150, 200], [120, 110, 170, 210]], [1, 2])
    # ID 1 cai: caixa deitada e deslocamento de 30 px
    return t.atualizar([[100, 160, 240, 220], [120, 110, 170, 210]], [1, 2])


def test_eventos_de_queda_e_proximidade_tem_ids():
    pessoas = _cena()
    interacoes = analisar_interacoes([p["centroide"] for p in pessoas], distancia_minima=200)
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    _, eventos = desenhar_anotacoes(frame, pessoas, interacoes, modo_velocidade=True, limiar_queda=15.0)

    tipos = {e["tipo"]: e for e in eventos}
    assert tipos["QUEDA"]["id"] == 1
    assert tipos["PROXIMIDADE"]["ids"] == (1, 2)
    assert tipos["PROXIMIDADE"]["postura"] == "Deitado e Em pe"


def test_modo_estatico_dispara_sem_velocidade():
    pessoas = _cena()
    for p in pessoas:
        p["velocidade"] = 0.0
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    _, com = desenhar_anotacoes(frame, pessoas, [], modo_velocidade=True)
    _, sem = desenhar_anotacoes(frame, pessoas, [], modo_velocidade=False)
    assert not any(e["tipo"] == "QUEDA" for e in com)
    assert any(e["tipo"] == "QUEDA" for e in sem)
