"""Métricas da avaliação por evento no conjunto do UR Fall e resultados pré-calculados dos 5 modelos.

Cada vídeo é um evento: uma queda é VP se o sistema registrar ao menos uma queda (senão FN) e
uma atividade diária é FP se houver queda registrada (senão VN), como na Tabela 2 do artigo.

Os resultados pré-calculados (`data/urfall_resultados.csv`) vêm da avaliação pelo CLI
(`tools/avaliar_urfall.py`, PyTorch, só a imagem colorida) com os parâmetros padrão da
metodologia. Só valem para esses parâmetros; com outros, a avaliação precisa rodar de verdade.

Este módulo não depende do Streamlit nem do OpenCV, para poder ser testado isoladamente.
"""
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

RAIZ = Path(__file__).resolve().parent.parent
CAMINHO_PRE_CALCULADOS = RAIZ / "data" / "urfall_resultados.csv"
CAMINHO_BENCHMARK = RAIZ / "data" / "tcc_benchmark_data.json"
ADL_DO_ARTIGO = 30  # o artigo usou as 30 primeiras atividades diárias (60 eventos)

# Parâmetros com que os resultados pré-calculados foram gerados. A distância de proximidade e o
# agrupamento em eventos não mudam se um vídeo tem ou não queda registrada, por isso não entram aqui.
PARAMETROS_PRE_CALCULADOS = {"conf": 0.5, "ar": 0.75, "vel": 15.0, "largura": 1080, "corrigir_fps": True,
                             "pular": 1}


def classificar(rotulo: str, quedas: int) -> str:
    """VP, VN, FP ou FN para um vídeo, segundo o rótulo esperado e o número de quedas registradas."""
    if rotulo == "queda":
        return "VP" if quedas > 0 else "FN"
    return "FP" if quedas > 0 else "VN"


def metricas(siglas: Iterable[str]) -> Dict[str, float]:
    """Matriz de confusão e métricas a partir das classificações dos vídeos já avaliados."""
    siglas = list(siglas)
    vn, fp, fn, vp = (siglas.count(s) for s in ("VN", "FP", "FN", "VP"))
    total = vn + fp + fn + vp
    precisao = vp / (vp + fp) if vp + fp else 0.0
    revocacao = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    return {"tn": vn, "fp": fp, "fn": fn, "tp": vp, "total": total, "acuracia": (vn + vp) / total if total else 0.0,
            "precisao": precisao, "recall": revocacao, "f1_score": f1}


def selecionar_videos(videos: List[Dict], n_adl: int = 40, somente_quedas: bool = False,
                      limite: Optional[int] = None) -> List[Dict]:
    """Quedas e as `n_adl` primeiras atividades diárias, na ordem do conjunto; `limite` corta a lista."""
    escolhidos = [v for v in videos if v["rotulo"] == "queda"
                  or (not somente_quedas and v.get("indice", 0) <= n_adl)]
    return escolhidos[:limite] if limite else escolhidos


def parametros_compativeis(**parametros) -> bool:
    """True se os parâmetros escolhidos são os usados nos resultados pré-calculados."""
    return all(parametros.get(k) == v for k, v in PARAMETROS_PRE_CALCULADOS.items())


def carregar_pre_calculados(caminho: Path = CAMINHO_PRE_CALCULADOS) -> Dict[str, Dict[str, Dict]]:
    """{modelo (ex.: 'yolo11s'): {id do vídeo: {'com': quedas, 'sem': quedas}}}."""
    resultados: Dict[str, Dict[str, Dict]] = {}
    with open(caminho, newline="", encoding="utf-8") as f:
        for linha in csv.DictReader(f):
            resultados.setdefault(linha["modelo"], {})[linha["video"]] = {
                "com": int(linha["quedas_com"]), "sem": int(linha["quedas_sem"])}
    return resultados


def linha_do_artigo(arquivo_modelo: str, cenario: str, caminho: Path = CAMINHO_BENCHMARK) -> Optional[Dict]:
    """Linha da Tabela 2 (UR Fall, 60 eventos) para o modelo ('yolo11s.pt') e cenário ('com' ou 'sem')."""
    nome = "YOLOv11" + Path(arquivo_modelo).stem[-1]
    rotulo = {"com": "Com Velocidade", "sem": "Sem Velocidade"}[cenario]
    dados = json.loads(Path(caminho).read_text(encoding="utf-8"))
    for linha in dados.get("tabela_2_ur_fall", []):
        if linha["modelo"].rstrip("*") == nome and linha["cenario"] == rotulo:
            return linha
    return None
