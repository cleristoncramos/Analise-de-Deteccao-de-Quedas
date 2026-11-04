"""Cache em disco dos resultados de análise das amostras do catálogo.

A análise é determinística para o mesmo vídeo, modelo, modo e parâmetros (o
rastreador é zerado a cada vídeo), então repetir uma análise idêntica pode
reaproveitar os eventos e o vídeo anotado já gerados.
"""
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple


def pasta_cache_resultados() -> Path:
    return Path(os.environ.get("DQ_CACHE_RESULTADOS") or Path(tempfile.gettempdir()) / "deteccao_quedas_resultados")


def chave_resultado(**parametros) -> str:
    texto = json.dumps(parametros, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:24]


def carregar_resultado(chave: str, pasta: Optional[Path] = None) -> Optional[Tuple[Dict, Optional[Path]]]:
    pasta = Path(pasta or pasta_cache_resultados()) / chave
    arquivo = pasta / "resultado.json"
    if not arquivo.is_file():
        return None
    try:
        dados = json.loads(arquivo.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    video = pasta / "video.mp4"
    return dados, (video if video.is_file() else None)


def salvar_resultado(chave: str, dados: Dict, video: Optional[Path] = None, pasta: Optional[Path] = None) -> Path:
    """Grava em uma pasta temporária e renomeia no fim, para nunca deixar um resultado pela metade."""
    base = Path(pasta or pasta_cache_resultados())
    base.mkdir(parents=True, exist_ok=True)
    trabalho = Path(tempfile.mkdtemp(dir=base, prefix=f".{chave}."))
    (trabalho / "resultado.json").write_text(json.dumps(dados, ensure_ascii=False), encoding="utf-8")
    if video and Path(video).is_file():
        shutil.copy2(video, trabalho / "video.mp4")
    destino = base / chave
    if destino.exists():
        shutil.rmtree(destino, ignore_errors=True)
    os.replace(trabalho, destino)
    return destino
