"""Detecção do ambiente de execução e "modo leve" para o Streamlit Community Cloud.

No plano gratuito do Streamlit Cloud, uso intenso e contínuo de CPU faz a plataforma
reduzir temporariamente a CPU do app (aviso "Your app has been throttled"). Rodar
o YOLO em todos os quadros, sobretudo com os modelos L e X, provoca isso. No Cloud o
app passa a usar padrões mais leves; fora dele, os padrões seguem a metodologia do TCC.
"""
import os
from dataclasses import dataclass
from pathlib import Path


def em_streamlit_cloud() -> bool:
    """O Streamlit Cloud monta o repositório em /mount/src e roda como 'adminuser'."""
    if os.environ.get("DQ_MODO_LEVE") is not None:  # permite forçar (1) ou desligar (0) o modo leve
        return os.environ["DQ_MODO_LEVE"] not in ("0", "", "false", "False")
    return Path("/mount/src").is_dir() or os.environ.get("HOME") == "/home/adminuser"


@dataclass(frozen=True)
class PadroesExecucao:
    modo_leve: bool
    pular_quadros: int          # processa 1 a cada N quadros
    max_quadros: int            # 0 = sem limite
    pausa_s: float              # pausa cooperativa após cada quadro processado
    largura_exibicao: int       # largura máxima da imagem enviada ao navegador
    threads_torch: int
    usar_openvino: bool         # inferência acelerada (eventos idênticos ao PyTorch no catálogo)


def _flag(nome: str, padrao: bool) -> bool:
    valor = os.environ.get(nome)
    return padrao if valor is None else valor not in ("0", "", "false", "False")


def padroes_execucao() -> PadroesExecucao:
    if em_streamlit_cloud():
        # Pular quadros fica desligado por padrão: dilui o pico de velocidade da queda e muda o resultado
        return PadroesExecucao(modo_leve=True, pular_quadros=1, max_quadros=900, pausa_s=0.02,
                               largura_exibicao=720, threads_torch=2, usar_openvino=_flag("DQ_OPENVINO", True))
    return PadroesExecucao(modo_leve=False, pular_quadros=1, max_quadros=0, pausa_s=0.0,
                           largura_exibicao=1080, threads_torch=0, usar_openvino=_flag("DQ_OPENVINO", False))


def configurar_torch(threads: int) -> None:
    """Limita as threads do PyTorch (0 mantém o padrão). Chamado uma vez na inicialização."""
    if threads <= 0:
        return
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass


MODELOS_PESADOS = ("v11l", "v11x")
