import os

import streamlit as st

# Importação dos módulos de página
from pages_modules.pagina_inicial import render_pagina_inicial
from pages_modules.informacoes_academicas import render_informacoes_academicas
from pages_modules.visao_geral import render_visao_geral
from pages_modules.estudio_inferencia import render_estudio_inferencia
from pages_modules.dashboard_metricas import render_dashboard_metricas
from pages_modules.auditoria_logs import render_auditoria_logs
from pages_modules.laboratorio_experimental import render_laboratorio_experimental
from pages_modules.navegacao import PAGINA_ACADEMICA, PAGINA_INICIAL, PAGINA_SISTEMA, ocultar_sidebar, pagina_atual, ir_para

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

MODULOS = {
    "Visão Geral e Artigo": render_visao_geral,
    "Estúdio de Inferência": render_estudio_inferencia,
    "Dashboard Científico": render_dashboard_metricas,
    "Auditoria e Logs": render_auditoria_logs,
    "Laboratório Experimental": render_laboratorio_experimental,
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

        menu = st.radio("Módulos do sistema", list(MODULOS), index=0)

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
