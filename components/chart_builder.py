import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

def criar_matriz_confusao(tn: int, fp: int, fn: int, tp: int, titulo: str = "Matriz de Confusão"):
    """
    Renderiza um heatmap de matriz de confusão elegante com Plotly.
    """
    z = [[tn, fp], [fn, tp]]
    x = ["Predito: Normal", "Predito: Queda"]
    y = ["Real: Normal", "Real: Queda"]
    text = [[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=x,
        y=y,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 16, "family": "Arial, sans-serif", "color": "white"},
        colorscale=[
            [0.0, "rgba(30, 41, 59, 0.8)"],
            [0.5, "rgba(59, 130, 246, 0.7)"],
            [1.0, "rgba(239, 68, 68, 0.85)"]
        ],
        showscale=False
    ))

    fig.update_layout(
        title={"text": titulo, "x": 0.5, "xanchor": "center", "font": {"size": 15}},
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8")
    )
    return fig

def criar_grafico_comparativo_metricas(df: pd.DataFrame, titulo: str = "Comparativo de Métricas por Modelo"):
    """
    Renderiza um gráfico de barras agrupadas comparando Acurácia, Precisão, Recall e F1-Score.
    """
    df_plot = df.copy()
    df_melt = pd.melt(
        df_plot,
        id_vars=["modelo", "cenario"],
        value_vars=["acuracia", "precisao", "recall", "f1_score"],
        var_name="Métrica",
        value_name="Valor"
    )
    df_melt["Rótulo"] = df_melt["modelo"] + " (" + df_melt["cenario"] + ")"
    mapa_metricas = {
        "acuracia": "Acurácia",
        "precisao": "Precisão (P)",
        "recall": "Revocação (Recall)",
        "f1_score": "F1-Score"
    }
    df_melt["Métrica"] = df_melt["Métrica"].map(mapa_metricas)

    fig = px.bar(
        df_melt,
        x="Rótulo",
        y="Valor",
        color="Métrica",
        barmode="group",
        title=titulo,
        color_discrete_sequence=["#38bdf8", "#34d399", "#fbbf24", "#f43f5e"]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis_title="",
        yaxis_title="Score (0 a 1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=50),
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(255,255,255,0.08)")
    )
    return fig

def criar_grafico_radar_metricas(dados_linha: Dict[str, Any], titulo: str = "Perfil do Modelo"):
    """
    Renderiza um gráfico de radar para um modelo específico.
    """
    categorias = ["Acurácia", "Precisão", "Recall", "F1-Score"]
    valores = [
        dados_linha.get("acuracia", 0),
        dados_linha.get("precisao", 0),
        dados_linha.get("recall", 0),
        dados_linha.get("f1_score", 0)
    ]
    # Fechar o círculo
    valores.append(valores[0])
    categorias.append(categorias[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=categorias,
        fill="toself",
        fillcolor="rgba(56, 189, 248, 0.25)",
        line=dict(color="#38bdf8", width=2),
        name=dados_linha.get("modelo", "Modelo")
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(255,255,255,0.1)"),
            bgcolor="rgba(0,0,0,0)"
        ),
        title={"text": titulo, "x": 0.5, "xanchor": "center", "font": {"size": 14}},
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        margin=dict(l=30, r=30, t=40, b=30),
        showlegend=False,
        height=300
    )
    return fig

def criar_grafico_reducao_falsos_positivos(df_geral: pd.DataFrame):
    """
    Destaca a redução drástica de Falsos Positivos entre Sem Velocidade e Com Velocidade.
    """
    fig = px.bar(
        df_geral,
        x="modelo",
        y="fp",
        color="cenario",
        barmode="group",
        title="Impacto do Limitador Cinemático na Redução de Falsos Positivos (FP)",
        color_discrete_map={
            "Com Velocidade": "#10b981",
            "Sem Velocidade": "#ef4444"
        },
        labels={"fp": "Falsos Positivos (FP)", "modelo": "Modelo Avaliado", "cenario": "Heurística"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        margin=dict(l=20, r=20, t=50, b=40)
    )
    return fig
