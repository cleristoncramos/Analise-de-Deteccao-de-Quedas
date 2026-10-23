import streamlit as st
from typing import Optional

def render_metric_card(titulo: str, valor: str, subtitulo: str = "", delta: Optional[str] = None, delta_color: str = "normal"):
    """
    Renderiza um card de métrica com cantos retos e borda fina.
    """
    card_html = f"""
    <div style="
        background: #fbfaf7;
        border: 1px solid #2b2b2b;
        padding: 14px 18px;
        margin-bottom: 12px;
    ">
        <div style="font-size: 0.78rem; color: #4a4a46; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{titulo}</div>
        <div style="font-family: 'Source Serif 4', Georgia, serif; font-size: 1.75rem; color: #1f3a5f; font-weight: 600; margin: 4px 0;">{valor}</div>
        <div style="font-size: 0.80rem; color: #4a4a46;">{subtitulo}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_alert_banner(tipo: str, mensagem: str):
    """
    Renderiza banners de notificação com estilo de alerta.
    """
    cores = {
        "QUEDA": {"bg": "rgba(123, 45, 38, 0.08)", "border": "#7b2d26", "text": "#5e211c", "rotulo": "Queda"},
        "PROXIMIDADE": {"bg": "rgba(154, 123, 47, 0.10)", "border": "#9a7b2f", "text": "#5f4b1b", "rotulo": "Proximidade"},
        "INFO": {"bg": "rgba(31, 58, 95, 0.07)", "border": "#1f3a5f", "text": "#1f3a5f", "rotulo": "Informação"},
        "SUCESSO": {"bg": "rgba(79, 109, 90, 0.10)", "border": "#4f6d5a", "text": "#34493c", "rotulo": "Sucesso"}
    }
    estilo = cores.get(tipo.upper(), cores["INFO"])
    banner_html = f"""
    <div style="
        background: {estilo['bg']};
        border: 1px solid {estilo['border']};
        border-left: 4px solid {estilo['border']};
        padding: 10px 14px;
        margin: 10px 0;
        color: {estilo['text']};
        font-size: 0.95rem;
    ">
        <b style="text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.06em; margin-right: 8px;">{estilo['rotulo']}</b>
        <span>{mensagem}</span>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)
