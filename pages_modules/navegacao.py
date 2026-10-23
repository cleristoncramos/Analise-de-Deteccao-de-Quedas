"""Controle de qual tela do app está aberta (página inicial, informações acadêmicas ou sistema)."""
import streamlit as st

PAGINA_INICIAL = "inicio"
PAGINA_ACADEMICA = "academico"
PAGINA_SISTEMA = "sistema"


def pagina_atual() -> str:
    return st.session_state.setdefault("pagina", PAGINA_INICIAL)


def ir_para(pagina: str):
    """Callback dos botões de navegação: o Streamlit reexecuta o script em seguida."""
    st.session_state["pagina"] = pagina


def ocultar_sidebar():
    st.markdown("""
    <style>
        section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
        [data-testid="stExpandSidebarButton"] { display: none !important; }
    </style>
    """, unsafe_allow_html=True)
