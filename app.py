import os

import streamlit as st

# Importação dos módulos de página
from pages_modules.visao_geral import render_visao_geral
from pages_modules.estudio_inferencia import render_estudio_inferencia
from pages_modules.dashboard_metricas import render_dashboard_metricas
from pages_modules.auditoria_logs import render_auditoria_logs
from pages_modules.laboratorio_experimental import render_laboratorio_experimental

# Configuração da página
st.set_page_config(
    page_title="Sistema de Detecção de Quedas | TCC UFPI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Carregamento do CSS customizado
css_path = os.path.join("assets", "style.css")
if os.path.isfile(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def main():
    # Barra lateral de navegação
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2 style="margin: 0; font-size: 1.35rem; color: #38bdf8;">🛡️ FallDetection AI</h2>
            <span style="font-size: 0.80rem; color: #94a3b8;">TCC - CSTGD / CEAD / UFPI</span>
        </div>
        """, unsafe_allow_html=True)

        menu = st.radio(
            "Navegação Principal",
            [
                "📄 Visão Geral & Artigo SBC",
                "🎥 Estúdio de Inferência",
                "📊 Dashboard Científico",
                "📋 Auditoria & Logs",
                "🧪 Laboratório Experimental"
            ],
            index=0
        )

        st.divider()
        st.markdown("""
        <div style="font-size: 0.75rem; color: #64748b; line-height: 1.4;">
            <b>Pesquisa:</b> PoC Prevenção de Acidentes<br>
            <b>Autores:</b> Brendo Soares, Wener Sampaio, Martony Silva<br>
            <b>Instituição:</b> Universidade Federal do Piauí
        </div>
        """, unsafe_allow_html=True)

    # Roteamento de páginas
    if menu == "📄 Visão Geral & Artigo SBC":
        render_visao_geral()
    elif menu == "🎥 Estúdio de Inferência":
        render_estudio_inferencia()
    elif menu == "📊 Dashboard Científico":
        render_dashboard_metricas()
    elif menu == "📋 Auditoria & Logs":
        render_auditoria_logs()
    elif menu == "🧪 Laboratório Experimental":
        render_laboratorio_experimental()

if __name__ == "__main__":
    main()
