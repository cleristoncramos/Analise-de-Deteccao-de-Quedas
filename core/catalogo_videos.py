"""Catálogo de vídeos de amostra do UR Fall Detection Dataset, baixados na hora do uso.

Os vídeos não são versionados: o catálogo (`videos/catalogo.json`) guarda só o
nome, a URL de origem e o rótulo esperado de cada um. Na primeira vez que uma
amostra é escolhida, o arquivo é baixado para um cache local e reaproveitado
nas próximas execuções.

Este módulo não depende do Streamlit nem do OpenCV, para poder ser testado isoladamente.
"""
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable, Dict, List, Optional

CAMINHO_CATALOGO = Path(__file__).resolve().parent.parent / "videos" / "catalogo.json"
ROTULOS = {"queda": "Queda", "nao_queda": "Sem queda"}
TEMPO_LIMITE_S = 60
TAMANHO_BLOCO = 256 * 1024


class ErroDownloadVideo(Exception):
    """O vídeo da amostra não pôde ser baixado ou o arquivo recebido não é um vídeo válido."""


def carregar_catalogo(caminho: Path = CAMINHO_CATALOGO) -> Dict:
    """Lê o catálogo e confere os campos obrigatórios de cada vídeo."""
    with open(caminho, "r", encoding="utf-8") as f:
        catalogo = json.load(f)
    for video in catalogo.get("videos", []):
        faltando = [c for c in ("id", "nome", "arquivo", "url", "rotulo") if not video.get(c)]
        if faltando:
            raise ValueError(f"Vídeo {video.get('id', '?')} do catálogo sem os campos: {', '.join(faltando)}")
        if video["rotulo"] not in ROTULOS:
            raise ValueError(f"Rótulo inválido no vídeo {video['id']}: {video['rotulo']}")
    return catalogo


def pasta_cache() -> Path:
    """Pasta onde os vídeos baixados ficam guardados (pode ser trocada pela variável DQ_CACHE_VIDEOS)."""
    return Path(os.environ.get("DQ_CACHE_VIDEOS") or Path(tempfile.gettempdir()) / "deteccao_quedas_videos")


def _baixar_url(url: str, destino, tempo_limite: float):
    """Copia o conteúdo da URL para o arquivo aberto `destino`, em blocos."""
    requisicao = urllib.request.Request(url, headers={"User-Agent": "deteccao-quedas-tcc"})
    with urllib.request.urlopen(requisicao, timeout=tempo_limite) as resposta:
        while True:
            bloco = resposta.read(TAMANHO_BLOCO)
            if not bloco:
                break
            destino.write(bloco)


def _parece_mp4(caminho: Path) -> bool:
    """Confere a assinatura do contêiner MP4 ('ftyp' logo no início do arquivo)."""
    try:
        with open(caminho, "rb") as f:
            cabecalho = f.read(12)
    except OSError:
        return False
    return len(cabecalho) == 12 and cabecalho[4:8] == b"ftyp"


def obter_video(video: Dict, pasta: Optional[Path] = None, baixar: Callable = _baixar_url,
                tempo_limite: float = TEMPO_LIMITE_S) -> Path:
    """Retorna o caminho local do vídeo, baixando-o se ainda não estiver no cache.

    O download vai para um arquivo temporário na mesma pasta e só é renomeado para o
    nome final depois de completo e validado, para que um download interrompido não
    deixe no cache um vídeo pela metade.
    """
    pasta = Path(pasta or pasta_cache())
    pasta.mkdir(parents=True, exist_ok=True)
    final = pasta / video["arquivo"]
    if final.is_file() and _parece_mp4(final):
        return final

    descritor, temporario = tempfile.mkstemp(dir=pasta, prefix=f".{video['arquivo']}.", suffix=".part")
    try:
        with os.fdopen(descritor, "wb") as destino:
            baixar(video["url"], destino, tempo_limite)
        if not _parece_mp4(Path(temporario)):
            raise ErroDownloadVideo(f"O arquivo recebido de {video['url']} não é um vídeo MP4 válido.")
        os.replace(temporario, final)
    except ErroDownloadVideo:
        raise
    except Exception as exc:  # rede fora do ar, tempo esgotado, HTTP 404 etc.
        raise ErroDownloadVideo(f"Não foi possível baixar {video['arquivo']} ({type(exc).__name__}: {exc}).") from exc
    finally:
        if os.path.exists(temporario):
            os.remove(temporario)
    return final


def recortar_quadro(quadro, recorte: Optional[str]):
    """O UR Fall grava profundidade e RGB lado a lado; 'direita' mantém só a metade RGB."""
    if recorte == "direita":
        return quadro[:, quadro.shape[1] // 2:]
    if recorte == "esquerda":
        return quadro[:, : quadro.shape[1] // 2]
    return quadro


def avaliar_resultado(rotulo: str, houve_queda: bool) -> str:
    """Classifica o resultado de um vídeo em VP, VN, FP ou FN segundo o rótulo esperado."""
    if rotulo == "queda":
        return "VP" if houve_queda else "FN"
    return "FP" if houve_queda else "VN"


def descrever_videos(catalogo: Dict) -> List[str]:
    return [f"{v['nome']} (esperado: {ROTULOS[v['rotulo']].lower()})" for v in catalogo.get("videos", [])]


def exemplos_recomendados(catalogo: Dict) -> List[Dict]:
    """Vídeos marcados com "destaque" no catálogo, na ordem indicada."""
    return sorted((v for v in catalogo.get("videos", []) if v.get("destaque")), key=lambda v: v["destaque"])
