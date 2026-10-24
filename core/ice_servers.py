"""Servidores ICE (STUN/TURN) para a câmera do navegador via WebRTC.

No Streamlit Cloud o servidor fica atrás de NAT e só STUN não basta para a
conexão de vídeo: é preciso um servidor TURN. As credenciais vêm da seção
[turn] dos secrets do Streamlit, nesta ordem de preferência:

1. Metered (metered_app + metered_api_key): credenciais temporárias pela API;
2. Twilio (twilio_account_sid + twilio_auth_token): token de Network Traversal;
3. ice_servers: lista fixa com urls, username e credential;
4. sem configuração (ou se a busca falhar): STUN público do Google.

Este módulo não depende do Streamlit, para poder ser testado isoladamente.
"""
import base64
import json
import urllib.parse
import urllib.request
from typing import Callable, List, Mapping, Optional, Tuple

STUN_GOOGLE = [{"urls": ["stun:stun.l.google.com:19302"]}]
TEMPO_LIMITE_S = 8


def _requisitar_json(url: str, dados: Optional[bytes] = None, cabecalhos: Optional[dict] = None):
    requisicao = urllib.request.Request(url, data=dados, headers=cabecalhos or {})
    with urllib.request.urlopen(requisicao, timeout=TEMPO_LIMITE_S) as resposta:
        return json.loads(resposta.read().decode("utf-8"))


def _metered(app: str, chave: str, requisitar) -> List[dict]:
    url = (f"https://{urllib.parse.quote(app)}.metered.live/api/v1/turn/credentials"
           f"?apiKey={urllib.parse.quote(chave)}")
    return requisitar(url)


def _twilio(sid: str, token: str, requisitar) -> List[dict]:
    autorizacao = base64.b64encode(f"{sid}:{token}".encode()).decode()
    resposta = requisitar(
        f"https://api.twilio.com/2010-04-01/Accounts/{urllib.parse.quote(sid)}/Tokens.json",
        dados=b"",
        cabecalhos={"Authorization": f"Basic {autorizacao}"},
    )
    return resposta["ice_servers"]


def _normalizar(servidores) -> List[dict]:
    """Aceita 'url' ou 'urls' e mantém só os campos que o navegador e o aiortc entendem."""
    normalizados = []
    for s in servidores or []:
        urls = s.get("urls") or s.get("url")
        if not urls:
            continue
        item = {"urls": urls if isinstance(urls, list) else [urls]}
        if s.get("username"):
            item["username"] = s["username"]
        if s.get("credential"):
            item["credential"] = s["credential"]
        normalizados.append(item)
    return normalizados


def montar_ice_servers(turn: Optional[Mapping], requisitar: Callable = _requisitar_json
                       ) -> Tuple[List[dict], str, Optional[str]]:
    """Retorna (ice_servers, origem, erro). `turn` é a seção [turn] dos secrets (ou None)."""
    turn = turn or {}
    erro = None
    try:
        if turn.get("metered_app") and turn.get("metered_api_key"):
            servidores = _normalizar(_metered(turn["metered_app"], turn["metered_api_key"], requisitar))
            if servidores:
                return servidores, "Metered", None
        elif turn.get("twilio_account_sid") and turn.get("twilio_auth_token"):
            servidores = _normalizar(_twilio(turn["twilio_account_sid"], turn["twilio_auth_token"], requisitar))
            if servidores:
                return servidores, "Twilio", None
        elif turn.get("ice_servers"):
            servidores = _normalizar([dict(s) for s in turn["ice_servers"]])
            if servidores:
                return servidores, "secrets", None
    except Exception as exc:  # rede fora do ar, chave inválida etc.: a página não pode cair
        erro = f"{type(exc).__name__}: {exc}"
    return STUN_GOOGLE, "STUN", erro


def tem_turn(servidores: List[dict]) -> bool:
    return any(str(u).startswith(("turn:", "turns:")) for s in servidores for u in s["urls"])
