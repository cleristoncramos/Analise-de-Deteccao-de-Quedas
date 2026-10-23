import streamlit as st
from typing import Optional

def render_metric_card(titulo: str, valor: str, subtitulo: str = "", delta: Optional[str] = None, delta_color: str = "normal"):
    """
    Renderiza um card visual de métrica com bordas arredondadas e estética refinada.
    """
    card_html = f"""
    <div style="
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        backdrop-filter: blur(8px);
    ">
        <div style="font-size: 0.85rem; color: #94a3b8; font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em;">{titulo}</div>
        <div style="font-size: 1.85rem; color: #f8fafc; font-weight: 700; margin: 4px 0;">{valor}</div>
        <div style="font-size: 0.80rem; color: #64748b;">{subtitulo}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_alert_banner(tipo: str, mensagem: str):
    """
    Renderiza banners de notificação com estilo de alerta.
    """
    cores = {
        "QUEDA": {"bg": "rgba(239, 68, 68, 0.15)", "border": "#ef4444", "text": "#fca5a5", "icon": "🚨"},
        "PROXIMIDADE": {"bg": "rgba(245, 158, 11, 0.15)", "border": "#f59e0b", "text": "#fcd34d", "icon": "⚠️"},
        "INFO": {"bg": "rgba(59, 130, 246, 0.15)", "border": "#3b82f6", "text": "#93c5fd", "icon": "ℹ️"},
        "SUCESSO": {"bg": "rgba(34, 197, 94, 0.15)", "border": "#22c55e", "text": "#86efac", "icon": "✅"}
    }
    estilo = cores.get(tipo.upper(), cores["INFO"])
    banner_html = f"""
    <div style="
        background: {estilo['bg']};
        border-left: 4px solid {estilo['border']};
        border-radius: 8px;
        padding: 12px 16px;
        margin: 10px 0;
        color: {estilo['text']};
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 10px;
    ">
        <span style="font-size: 1.25rem;">{estilo['icon']}</span>
        <span>{mensagem}</span>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)
