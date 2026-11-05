"""Avalia o pipeline do CLI nos vídeos do UR Fall Detection Dataset e compara com a Tabela 2 do artigo.

Para cada modelo, cada vídeo passa uma vez pelo YOLO (com a mesma função `analisar_video`
de core/analise_video.py, a mesma do cli.py) e é avaliado nos dois modos: com velocidade (proposta) e sem velocidade (linha
de base). Um vídeo de queda é TP se houver ao menos uma queda registrada (senão FN); uma
atividade de vida diária (ADL) é FP se houver queda registrada (senão TN).

Os vídeos (câmera 0, paralela ao chão) são baixados para o cache local na primeira vez e a
análise usa só a metade colorida de cada vídeo (--RECORTE direita), como na demo.

Saída em relatorios/urfall/<execução>/ (fora do git):
  - eventos/<modelo>/<video>_<modo>.csv: ocorrências de cada vídeo;
  - videos.csv: uma linha por vídeo, modelo e modo, com o número de quedas e a classificação;
  - resumo.csv e comparacao.md: matriz de confusão e métricas, com 30 ADL (adl-01..30) e com todas
    as 40, lado a lado com a Tabela 2 de data/tcc_benchmark_data.json.

Uso:
    python tools/avaliar_urfall.py --MODELOS yolo11s.pt
    python tools/avaliar_urfall.py --MODELOS yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt
    python tools/avaliar_urfall.py --MODELOS yolo11s.pt --RECORTE nenhum      # quadro inteiro (diagnóstico)
"""
import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from core.catalogo_videos import ErroDownloadVideo, obter_video  # noqa: E402

URL_BASE = "https://fenix.ur.edu.pl/mkepski/ds/data/"
NOME_MODELO = {"yolo11n": "YOLOv11n", "yolo11s": "YOLOv11s", "yolo11m": "YOLOv11m", "yolo11l": "YOLOv11l",
               "yolo11x": "YOLOv11x"}
CENARIO = {"com": "Com Velocidade", "sem": "Sem Velocidade"}


def lista_videos(n_quedas: int = 30, n_adl: int = 40) -> List[Dict]:
    videos = []
    for prefixo, total, rotulo in (("fall", n_quedas, "queda"), ("adl", n_adl, "nao_queda")):
        for i in range(1, total + 1):
            arquivo = f"{prefixo}-{i:02d}-cam0.mp4"
            videos.append({"id": f"{prefixo}-{i:02d}", "nome": arquivo, "arquivo": arquivo,
                           "url": URL_BASE + arquivo, "rotulo": rotulo, "indice": i})
    return videos


def classificar(rotulo: str, quedas: int) -> str:
    if rotulo == "queda":
        return "TP" if quedas > 0 else "FN"
    return "FP" if quedas > 0 else "TN"


def metricas(contagem: Dict[str, int]) -> Dict[str, float]:
    tn, fp, fn, tp = (contagem.get(k, 0) for k in ("TN", "FP", "FN", "TP"))
    total = tn + fp + fn + tp
    precisao = tp / (tp + fp) if tp + fp else 0.0
    revocacao = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precisao * revocacao / (precisao + revocacao) if precisao + revocacao else 0.0
    return {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "acuracia": (tn + tp) / total if total else 0.0,
            "precisao": precisao, "recall": revocacao, "f1_score": f1}


def resumir(linhas: List[Dict], max_adl: Optional[int]) -> Dict[tuple, Dict]:
    """Agrupa por (modelo, modo); com max_adl=30 usa só adl-01..30, como no artigo (60 eventos)."""
    contagens: Dict[tuple, Dict[str, int]] = {}
    for linha in linhas:
        if linha["rotulo"] == "nao_queda" and max_adl and linha["indice"] > max_adl:
            continue
        chave = (linha["modelo"], linha["modo"])
        contagens.setdefault(chave, {}).setdefault(linha["classe"], 0)
        contagens[chave][linha["classe"]] += 1
    return {chave: metricas(c) for chave, c in contagens.items()}


def tabela_artigo() -> Dict[tuple, Dict]:
    dados = json.loads((RAIZ / "data" / "tcc_benchmark_data.json").read_text(encoding="utf-8"))
    return {(r["modelo"].rstrip("*"), r["cenario"]): r for r in dados["tabela_2_ur_fall"]}


def comparacao_markdown(resumo30: Dict, resumo40: Dict, artigo: Dict, cabecalho: str) -> str:
    linhas = [cabecalho, "",
              "| Modelo | Cenário | Artigo (TN/FP/FN/TP) | CLI 60 eventos (TN/FP/FN/TP) | Diferença | "
              "Acurácia artigo → CLI | CLI 70 eventos (40 ADL) |",
              "|---|---|---|---|---|---|---|"]
    for (modelo, modo), m in sorted(resumo30.items()):
        nome, cenario = NOME_MODELO.get(modelo, modelo), CENARIO[modo]
        ref = artigo.get((nome, cenario))
        cli = f"{m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}"
        m40 = resumo40.get((modelo, modo))
        cli40 = f"{m40['tn']}/{m40['fp']}/{m40['fn']}/{m40['tp']}" if m40 else "—"
        if ref:
            art = f"{ref['tn']}/{ref['fp']}/{ref['fn']}/{ref['tp']}"
            dif = "/".join(f"{m[k] - ref[k]:+d}" for k in ("tn", "fp", "fn", "tp"))
            acc = f"{ref['acuracia']:.3f} → {m['acuracia']:.3f}"
        else:
            art, dif, acc = "não avaliado", "—", f"— → {m['acuracia']:.3f}"
        linhas.append(f"| {nome} | {cenario} | {art} | {cli} | {dif} | {acc} | {cli40} |")
    return "\n".join(linhas) + "\n"


def avaliar(modelos: List[str], recorte: str, rapido: bool, saida: Path, n_adl: int = 40,
            carregar: Optional[Callable] = None, abrir: Optional[Callable] = None,
            baixar: Optional[Callable] = None) -> Dict:
    """Roda a avaliação. `carregar`, `abrir` e `baixar` permitem substituir o YOLO, o OpenCV e a rede nos testes."""
    import cv2

    from core.analise_video import analisar_video
    from core.yolo_engine import carregar_modelo

    carregar = carregar or (lambda nome: carregar_modelo(nome, acelerar=rapido))
    abrir = abrir or cv2.VideoCapture
    videos = lista_videos(n_adl=n_adl)
    (saida / "eventos").mkdir(parents=True, exist_ok=True)

    caminhos = {}
    for video in videos:
        try:
            caminhos[video["id"]] = obter_video(video, baixar=baixar) if baixar else obter_video(video)
        except ErroDownloadVideo as exc:
            print(f"[AVISO] {exc}", flush=True)

    # Retoma uma execução anterior na mesma pasta: modelos já avaliados não são repetidos
    linhas = _ler_linhas(saida / "videos.csv")
    concluidos = {linha["modelo"] for linha in linhas}
    inicio = time.time()
    for nome_modelo in modelos:
        chave_modelo = Path(nome_modelo).stem
        if chave_modelo in concluidos:
            print(f"[{chave_modelo}] já avaliado em {saida}, pulando", flush=True)
            continue
        modelo = None
        pasta_eventos = saida / "eventos" / chave_modelo
        pasta_eventos.mkdir(parents=True, exist_ok=True)
        for n, video in enumerate(videos, start=1):
            if video["id"] not in caminhos:
                continue
            # Retoma vídeo a vídeo: os CSVs de eventos só são gravados quando o vídeo termina
            prontos = [pasta_eventos / f"{video['id']}_{modo}.csv" for modo in ("com", "sem")]
            if all(arq.is_file() for arq in prontos):
                for modo, arq in zip(("com", "sem"), prontos):
                    quedas = _contar_quedas(arq)
                    linhas.append({"modelo": chave_modelo, "modo": modo, "video": video["id"],
                                   "rotulo": video["rotulo"], "indice": video["indice"], "frames": 0,
                                   "quedas": quedas, "classe": classificar(video["rotulo"], quedas)})
                print(f"[{chave_modelo}] {n}/{len(videos)} {video['id']}: já analisado, reaproveitado", flush=True)
                continue
            if modelo is None:
                modelo = carregar(nome_modelo)
            t0 = time.time()
            resultado = analisar_video(abrir(str(caminhos[video["id"]])), modelo, modos=("com", "sem"),
                                       recorte=recorte, janela=False)
            for modo in ("com", "sem"):
                with open(pasta_eventos / f"{video['id']}_{modo}.csv", "w", newline="", encoding="utf-8") as f:
                    escritor = csv.writer(f)
                    escritor.writerow(["Frame", "Data_Hora", "Tipo_Evento", "Detalhes", "Postura_Inferida",
                                       "Velocidade"])
                    escritor.writerows(resultado["eventos"][modo])
                quedas = resultado["quedas"][modo]
                linhas.append({"modelo": chave_modelo, "modo": modo, "video": video["id"], "rotulo": video["rotulo"],
                               "indice": video["indice"], "frames": resultado["frames"], "quedas": quedas,
                               "classe": classificar(video["rotulo"], quedas)})
            print(f"[{chave_modelo}] {n}/{len(videos)} {video['id']}: {resultado['frames']} quadros em "
                  f"{time.time() - t0:.0f} s | com: {resultado['quedas']['com']} queda(s), "
                  f"sem: {resultado['quedas']['sem']} queda(s) | total {time.time() - inicio:.0f} s", flush=True)
        _gravar(linhas, saida, recorte, rapido)
    return {"linhas": linhas, "resumo30": resumir(linhas, 30), "resumo40": resumir(linhas, None)}


def _contar_quedas(arquivo: Path) -> int:
    with open(arquivo, newline="", encoding="utf-8") as f:
        return sum(1 for linha in csv.DictReader(f) if linha["Tipo_Evento"] == "QUEDA")


def _impedir_suspensao():
    """No Windows, pede ao sistema que não suspenda enquanto este processo roda (vale só para o processo)."""
    if sys.platform == "win32":
        try:
            import ctypes

            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        except Exception:
            pass


def _ler_linhas(arquivo: Path) -> List[Dict]:
    if not arquivo.is_file():
        return []
    with open(arquivo, newline="", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))
    for linha in linhas:
        for campo in ("indice", "frames", "quedas"):
            linha[campo] = int(linha[campo])
    return linhas


def _gravar(linhas: List[Dict], saida: Path, recorte: str, rapido: bool):
    with open(saida / "videos.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    resumo30, resumo40 = resumir(linhas, 30), resumir(linhas, None)
    with open(saida / "resumo.csv", "w", newline="", encoding="utf-8") as f:
        escritor = csv.writer(f)
        escritor.writerow(["conjunto", "modelo", "modo", "tn", "fp", "fn", "tp", "acuracia", "precisao", "recall",
                           "f1_score"])
        for conjunto, resumo in (("60 eventos (adl-01..30)", resumo30), ("70 eventos (40 ADL)", resumo40)):
            for (modelo, modo), m in sorted(resumo.items()):
                escritor.writerow([conjunto, modelo, modo, m["tn"], m["fp"], m["fn"], m["tp"], f"{m['acuracia']:.3f}",
                                   f"{m['precisao']:.3f}", f"{m['recall']:.3f}", f"{m['f1_score']:.3f}"])
    cabecalho = (f"# UR Fall: CLI × Tabela 2 do artigo\n\nRecorte: {recorte} · inferência: "
                 f"{'OpenVINO' if rapido else 'PyTorch'} · largura 1080 · AR < 0,75 · v ≥ 15 px/f · carência 2 s · "
                 f"gerado em {datetime.now():%d/%m/%Y %H:%M}")
    (saida / "comparacao.md").write_text(comparacao_markdown(resumo30, resumo40, tabela_artigo(), cabecalho),
                                         encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Avalia o CLI no UR Fall Detection Dataset e compara com a Tabela 2")
    parser.add_argument("--MODELOS", nargs="+", default=["yolo11s.pt"])
    parser.add_argument("--RECORTE", choices=["direita", "esquerda", "nenhum"], default="direita")
    parser.add_argument("--RAPIDO", action="store_true", help="Usa OpenVINO em vez de PyTorch")
    parser.add_argument("--ADL", type=int, default=40, help="Quantas ADL avaliar (o dataset tem 40)")
    parser.add_argument("--SAIDA", default=None, help="Pasta de saída (padrão: relatorios/urfall/<data_hora>)")
    args = parser.parse_args(argv)
    saida = Path(args.SAIDA or RAIZ / "relatorios" / "urfall" / f"{datetime.now():%Y%m%d_%H%M}_{args.RECORTE}")
    _impedir_suspensao()
    avaliar(args.MODELOS, args.RECORTE, args.RAPIDO, saida, n_adl=args.ADL)
    print(f"[OK] Resultados em {saida}")
    print((saida / "comparacao.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
