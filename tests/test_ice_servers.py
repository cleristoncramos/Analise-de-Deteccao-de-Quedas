"""Montagem da configuração STUN/TURN da câmera do navegador a partir de secrets simulados."""
from core.ice_servers import STUN_GOOGLE, montar_ice_servers, tem_turn

TURN_METERED = [
    {"urls": "stun:stun.relay.metered.ca:80"},
    {"urls": "turn:global.relay.metered.ca:80", "username": "u", "credential": "c"},
]


def test_sem_secrets_usa_stun_google():
    servidores, origem, erro = montar_ice_servers(None)
    assert servidores == STUN_GOOGLE
    assert origem == "STUN" and erro is None
    assert not tem_turn(servidores)


def test_metered_busca_credenciais_pela_api():
    urls = []

    def requisitar(url, **_):
        urls.append(url)
        return TURN_METERED

    servidores, origem, erro = montar_ice_servers({"metered_app": "meuapp", "metered_api_key": "k"}, requisitar)
    assert origem == "Metered" and erro is None
    assert urls == ["https://meuapp.metered.live/api/v1/turn/credentials?apiKey=k"]
    assert servidores[1] == {"urls": ["turn:global.relay.metered.ca:80"], "username": "u", "credential": "c"}
    assert tem_turn(servidores)


def test_twilio_le_ice_servers_do_token():
    def requisitar(url, dados=None, cabecalhos=None):
        assert url.endswith("/Accounts/AC123/Tokens.json")
        assert cabecalhos["Authorization"].startswith("Basic ")
        return {"ice_servers": [{"url": "turn:global.turn.twilio.com:3478", "username": "u", "credential": "c"}]}

    servidores, origem, _ = montar_ice_servers({"twilio_account_sid": "AC123", "twilio_auth_token": "t"}, requisitar)
    assert origem == "Twilio"
    assert servidores == [{"urls": ["turn:global.turn.twilio.com:3478"], "username": "u", "credential": "c"}]


def test_lista_fixa_dos_secrets():
    turn = {"ice_servers": [{"urls": ["turn:turn.exemplo.org:3478"], "username": "u", "credential": "c"}]}
    servidores, origem, _ = montar_ice_servers(turn)
    assert origem == "secrets" and tem_turn(servidores)


def test_falha_na_busca_cai_no_stun_com_erro():
    def requisitar(url, **_):
        raise OSError("sem rede")

    servidores, origem, erro = montar_ice_servers({"metered_app": "a", "metered_api_key": "k"}, requisitar)
    assert servidores == STUN_GOOGLE and origem == "STUN"
    assert "sem rede" in erro
