"""Inferência acelerada do YOLOv11 na CPU com OpenVINO.

No Streamlit Cloud a inferência do YOLO em PyTorch é o gargalo da análise (mais
de 90% do tempo por quadro). O mesmo modelo exportado para OpenVINO roda cerca de
duas vezes mais rápido na CPU. A exportação usa entrada dinâmica (`dynamic=True`)
para manter o mesmo pré-processamento retangular do PyTorch: nos oito vídeos do
catálogo do UR Fall os eventos detectados foram idênticos aos do PyTorch.

A exportação é feita uma vez por servidor e guardada em cache; se o OpenVINO não
estiver instalado ou a exportação falhar, o sistema volta ao PyTorch.
"""
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Optional

_TRAVA = threading.Lock()


def pasta_modelos_acelerados() -> Path:
    return Path(os.environ.get("DQ_CACHE_MODELOS") or Path(tempfile.gettempdir()) / "deteccao_quedas_modelos")


def openvino_disponivel() -> bool:
    try:
        import openvino  # noqa: F401
    except Exception:
        return False
    return True


def exportar_openvino(caminho_pesos: str, pasta: Optional[Path] = None) -> Optional[str]:
    """Retorna a pasta do modelo OpenVINO (exportando na primeira vez) ou None se não for possível."""
    if not openvino_disponivel():
        return None
    pasta = Path(pasta or pasta_modelos_acelerados())
    nome = Path(caminho_pesos).stem
    # A Ultralytics reconhece o formato pelo sufixo "_openvino_model" da pasta
    destino = pasta / f"{nome}_dinamico_openvino_model"
    if (destino / f"{nome}.xml").is_file():
        return str(destino)

    with _TRAVA:  # duas sessões não exportam o mesmo modelo ao mesmo tempo
        if (destino / f"{nome}.xml").is_file():
            return str(destino)
        try:
            from ultralytics import YOLO

            pasta.mkdir(parents=True, exist_ok=True)
            trabalho = Path(tempfile.mkdtemp(dir=pasta))
            modelo = YOLO(caminho_pesos)
            pesos = Path(modelo.ckpt_path or caminho_pesos)
            shutil.copy2(pesos, trabalho / pesos.name)
            exportado = Path(YOLO(str(trabalho / pesos.name)).export(format="openvino", dynamic=True, half=False,
                                                                     verbose=False))
            if destino.exists():
                shutil.rmtree(destino, ignore_errors=True)
            shutil.move(str(exportado), str(destino))
            shutil.rmtree(trabalho, ignore_errors=True)
            return str(destino)
        except Exception:
            return None
