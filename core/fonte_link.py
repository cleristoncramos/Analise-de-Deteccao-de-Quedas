"""Vídeo a partir de um link: arquivo direto, transmissão (HLS, RTSP, RTMP, HTTP) ou YouTube.

Links diretos e transmissões são abertos pelo OpenCV (FFmpeg), quadro a quadro.
Links do YouTube são baixados antes com o yt-dlp, num formato MP4 de arquivo único
e com duração limitada. Servidores em nuvem costumam ser bloqueados pelo YouTube;
nesse caso a mensagem orienta a baixar o vídeo e usar o upload.

Este módulo não depende do Streamlit, para poder ser testado isoladamente.
"""
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

ESQUEMAS_ACEITOS = ("http", "https", "rtsp", "rtsps", "rtmp", "rtmps")
DOMINIOS_YOUTUBE = ("youtube.com", "youtu.be", "youtube-nocookie.com")


class ErroLink(Exception):
    """O link não pôde ser usado; a mensagem é exibida ao usuário."""


def validar_link(url: str) -> str:
    url = (url or "").strip()
    if not url:
        raise ErroLink("Informe o endereço do vídeo ou da transmissão.")
    partes = urlparse(url)
    if partes.scheme.lower() not in ESQUEMAS_ACEITOS or not partes.netloc:
        raise ErroLink("Endereço inválido. Use um link que comece com http://, https://, rtsp:// ou rtmp://.")
    return url


def eh_youtube(url: str) -> bool:
    dominio = urlparse(url).netloc.lower().split(":")[0]
    return any(dominio == d or dominio.endswith("." + d) for d in DOMINIOS_YOUTUBE)


def _baixar_com_ytdlp(url: str, pasta: Path, max_duracao_s: int) -> Path:
    import yt_dlp

    opcoes = {
        # A análise não usa áudio: basta a trilha de vídeo (H.264 em MP4, que o OpenCV lê), sem juntar com o som
        "format": "bv*[ext=mp4][vcodec^=avc1][height<=720]/bv*[ext=mp4][height<=720]/b[ext=mp4][height<=720]/bv*/b",
        "outtmpl": str(pasta / "youtube.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "match_filter": yt_dlp.utils.match_filter_func(f"duration <= {int(max_duracao_s)}"),
    }
    with yt_dlp.YoutubeDL(opcoes) as ydl:
        info = ydl.extract_info(url, download=True)
        if info is None or info.get("_type") == "playlist":
            raise ErroLink("O link aponta para uma playlist; informe o link de um único vídeo.")
        caminho = Path(ydl.prepare_filename(info))
    if not caminho.is_file():
        raise ErroLink(f"O vídeo tem mais de {max_duracao_s // 60} minuto(s) ou não pôde ser baixado.")
    return caminho


def baixar_youtube(url: str, pasta: Path, max_duracao_s: int = 180,
                   baixar: Optional[Callable] = None) -> Path:
    """Baixa o vídeo do YouTube para `pasta` e retorna o caminho, ou levanta ErroLink com uma explicação."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    try:
        return (baixar or _baixar_com_ytdlp)(url, pasta, max_duracao_s)
    except ErroLink:
        raise
    except ImportError as exc:
        raise ErroLink("O suporte a links do YouTube não está instalado (pacote yt-dlp).") from exc
    except Exception as exc:
        texto = str(exc)
        if any(t in texto.lower() for t in ("sign in", "bot", "429", "403", "confirm you")):
            raise ErroLink("O YouTube bloqueou o download a partir deste servidor, algo comum em servidores na "
                           "nuvem. Baixe o vídeo no seu computador e use a opção Upload de Arquivo de Vídeo.") from exc
        raise ErroLink(f"Não foi possível baixar o vídeo do YouTube ({texto.splitlines()[0][:200]}).") from exc
