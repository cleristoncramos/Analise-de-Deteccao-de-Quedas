import os

import streamlit as st

# Importação dos módulos de página
from pages_modules.pagina_inicial import render_pagina_inicial
from pages_modules.informacoes_academicas import render_informacoes_academicas
from pages_modules.visao_geral import render_visao_geral
from pages_modules.estudio_inferencia import render_estudio_inferencia
from pages_modules.auditoria_logs import render_auditoria_logs
from pages_modules.avaliacao_conjunto import render_avaliacao_conjunto
from pages_modules.constantes import (
    CHAVE_MODULO,
    MODULO_AUDITORIA,
    MODULO_CONJUNTO,
    MODULO_ESTUDIO,
    MODULO_VISAO_GERAL,
)
from pages_modules.navegacao import PAGINA_ACADEMICA, PAGINA_INICIAL, PAGINA_SISTEMA, ocultar_sidebar, pagina_atual, ir_para

from core.ambiente import configurar_torch, padroes_execucao

configurar_torch(padroes_execucao().threads_torch)

# Configuração da página
st.set_page_config(
    page_title="Sistema de Detecção de Quedas | TCC UFPI",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregamento do CSS customizado
css_path = os.path.join("assets", "style.css")
if os.path.isfile(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# A demo destaca a ferramenta; os resultados do artigo (tabelas e trabalhos futuros) ficam no README
MODULOS = {
    MODULO_ESTUDIO: render_estudio_inferencia,
    MODULO_CONJUNTO: render_avaliacao_conjunto,
    MODULO_AUDITORIA: render_auditoria_logs,
    MODULO_VISAO_GERAL: render_visao_geral,
}


def render_sistema():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-marca">
            <div class="titulo">Sistema de Análise e Detecção de Quedas</div>
            <div class="sub">TCC · CSTGD · CEAD · UFPI</div>
        </div>
        """, unsafe_allow_html=True)

        st.button("Sair para a página inicial", icon=":material/home:", on_click=ir_para,
                  args=(PAGINA_INICIAL,), width="stretch", key="botao_sair")

        if st.session_state.get(CHAVE_MODULO) not in MODULOS:
            st.session_state[CHAVE_MODULO] = MODULO_ESTUDIO
        menu = st.radio("Módulos do sistema", list(MODULOS), key=CHAVE_MODULO)

    MODULOS[menu]()


def main():
    pagina = pagina_atual()
    if pagina == PAGINA_SISTEMA:
        render_sistema()
    else:
        ocultar_sidebar()
        if pagina == PAGINA_ACADEMICA:
            render_informacoes_academicas()
        else:
            render_pagina_inicial()


if __name__ == "__main__":
    main()
