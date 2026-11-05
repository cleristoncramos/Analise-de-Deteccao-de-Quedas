"""Controle de qual tela do app está aberta (página inicial, informações acadêmicas ou sistema)."""
import streamlit as st

from pages_modules.constantes import (
    AMOSTRA_EXEMPLO,
    CHAVE_AMOSTRA_PADRAO,
    CHAVE_FONTE,
    CHAVE_MODO,
    CHAVE_MODULO,
    FONTE_AMOSTRA,
    MODO_AB,
    MODULO_ESTUDIO,
)

PAGINA_INICIAL = "inicio"
PAGINA_ACADEMICA = "academico"
PAGINA_SISTEMA = "sistema"


def pagina_atual() -> str:
    return st.session_state.setdefault("pagina", PAGINA_INICIAL)


def ir_para(pagina: str):
    """Callback dos botões de navegação: o Streamlit reexecuta o script em seguida."""
    st.session_state["pagina"] = pagina


def abrir_estudio_com_exemplo():
    """Callback do Get Started: abre o Estúdio já com uma amostra que o sistema acerta, no modo A/B."""
    st.session_state["pagina"] = PAGINA_SISTEMA
    st.session_state[CHAVE_MODULO] = MODULO_ESTUDIO
    st.session_state[CHAVE_FONTE] = FONTE_AMOSTRA
    st.session_state[CHAVE_MODO] = MODO_AB
    st.session_state[CHAVE_AMOSTRA_PADRAO] = AMOSTRA_EXEMPLO


def ocultar_sidebar():
    st.markdown("""
    <style>
        section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
        [data-testid="stExpandSidebarButton"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
