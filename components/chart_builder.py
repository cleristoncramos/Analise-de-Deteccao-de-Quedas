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
        textfont={"size": 16, "family": "Georgia, serif", "color": "#1d1d1b"},
        colorscale=[
            [0.0, "#f3efe4"],
            [0.5, "#b9c6d4"],
            [1.0, "#7f98b5"]
        ],
        xgap=1,
        ygap=1,
        showscale=False
    ))

    fig.update_layout(
        title={"text": titulo, "x": 0.5, "xanchor": "center", "font": {"size": 15}},
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1d1d1b")
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
        color_discrete_sequence=["#1f3a5f", "#4f6d5a", "#9a7b2f", "#7b2d26"]
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1d1d1b"),
        xaxis_title="",
        yaxis_title="Score (0 a 1)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        margin=dict(l=20, r=20, t=100, b=50),
        yaxis=dict(range=[0, 1.05], gridcolor="rgba(29,29,27,0.15)")
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
        fillcolor="rgba(31, 58, 95, 0.18)",
        line=dict(color="#1f3a5f", width=2),
        name=dados_linha.get("modelo", "Modelo")
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(29,29,27,0.15)"),
            bgcolor="rgba(0,0,0,0)"
        ),
        title={"text": titulo, "x": 0.5, "xanchor": "center", "font": {"size": 14}},
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1d1d1b"),
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
        title="Impacto do Limitador Cinemático<br>na Redução de Falsos Positivos (FP)",
        color_discrete_map={
            "Com Velocidade": "#4f6d5a",
            "Sem Velocidade": "#7b2d26"
        },
        labels={"fp": "Falsos Positivos (FP)", "modelo": "Modelo Avaliado", "cenario": "Heurística"}
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1d1d1b"),
        yaxis=dict(gridcolor="rgba(29,29,27,0.15)"),
        margin=dict(l=20, r=20, t=80, b=40)
    )
    return fig

# Paleta do app (assets/style.css)
_TINTA, _TINTA_SUAVE = "#1d1d1b", "#4a4a46"
_MARINHO, _VINHO, _VERDE = "#1f3a5f", "#7b2d26", "#4f6d5a"
_FONTE = "IBM Plex Sans, Helvetica, Arial, sans-serif"


def _num(valor: float) -> str:
    return f"{valor:g}".replace(".", ",")


def criar_grafico_movimento(serie: Dict[str, Any], limiar_ar: float = 0.75, limiar_vel: float = 15.0,
                            altura: int = 250):
    """
    Razão de aspecto e velocidade da pessoa principal ao longo dos quadros, em dois painéis
    alinhados: as faixas sombreadas marcam a região da regra (deitado; velocidade de queda),
    as linhas tracejadas os limiares e as faixas verticais os quadros em que a queda foi registrada.
    """
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    fig.add_trace(go.Scatter(
        x=serie["frame"], y=serie["ar"], mode="lines", name="Razão de aspecto",
        line=dict(color=_MARINHO, width=1.8, shape="spline", smoothing=0.4),
        fill="tozeroy", fillcolor="rgba(31, 58, 95, 0.07)",
        hovertemplate="AR %{y:.2f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=serie["frame"], y=serie["v"], mode="lines", name="Velocidade",
        line=dict(color=_VERDE, width=1.8, shape="spline", smoothing=0.4),
        fill="tozeroy", fillcolor="rgba(79, 109, 90, 0.08)",
        hovertemplate="v %{y:.1f} px/quadro<extra></extra>"), row=2, col=1)

    # Região em que cada ramo da regra é satisfeito, bem discreta, e o limiar tracejado com rótulo pequeno
    v_max = max([limiar_vel * 1.25] + [x for x in serie["v"] if x is not None])
    fig.add_hrect(y0=0, y1=limiar_ar, row=1, col=1, fillcolor=_VINHO, opacity=0.05, line_width=0, layer="below")
    fig.add_hrect(y0=limiar_vel, y1=v_max * 1.05, row=2, col=1, fillcolor=_VINHO, opacity=0.05, line_width=0,
                  layer="below")
    for linha, valor, texto in ((1, limiar_ar, f"deitado &lt; {_num(limiar_ar)}"),
                                (2, limiar_vel, f"queda ≥ {_num(limiar_vel)} px/quadro")):
        fig.add_hline(y=valor, row=linha, col=1, line=dict(color=_VINHO, width=1, dash="dot"),
                      annotation=dict(text=texto, font=dict(size=9.5, color=_VINHO), xanchor="right", x=1,
                                      yanchor="bottom", showarrow=False))

    # Quadros com queda registrada: faixa estreita e linha fina nos dois painéis, com rótulo no topo
    quadros = [x for x in serie["frame"] if x is not None]
    meia_faixa = max(0.6, (max(quadros) - min(quadros)) * 0.006) if quadros else 0.6
    for quadro in serie.get("quedas", []):
        for linha in (1, 2):
            fig.add_vrect(x0=quadro - meia_faixa, x1=quadro + meia_faixa, row=linha, col=1, fillcolor=_VINHO,
                          opacity=0.14, line_width=0)
            fig.add_vline(x=quadro, row=linha, col=1, line=dict(color=_VINHO, width=1.2))
        fig.add_annotation(x=quadro, y=1, yref="y domain", xref="x", text="queda", showarrow=False,
                           yanchor="bottom", font=dict(size=9.5, color=_VINHO), row=1, col=1)

    fig.update_layout(
        height=altura,
        margin=dict(l=8, r=8, t=18, b=6),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONTE, color=_TINTA_SUAVE, size=10.5),
        separators=",.",
        showlegend=False,
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#fbfaf7", bordercolor=_TINTA, font=dict(family=_FONTE, size=11, color=_TINTA)),
    )
    eixo = dict(showline=True, linecolor="rgba(29, 29, 27, 0.45)", linewidth=1, ticks="outside", ticklen=3,
                tickcolor="rgba(29, 29, 27, 0.45)", zeroline=False, title_font=dict(size=10, color=_TINTA))
    fig.update_xaxes(**eixo, showgrid=False, showspikes=True, spikemode="across", spikethickness=1,
                     spikecolor="rgba(29, 29, 27, 0.35)", spikedash="solid")
    fig.update_yaxes(**eixo, gridcolor="rgba(29, 29, 27, 0.08)", griddash="dot", nticks=4, rangemode="tozero")
    fig.update_yaxes(title_text="AR (h/w)", row=1, col=1)
    fig.update_yaxes(title_text="v (px/quadro)", range=[0, v_max * 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Quadro", row=2, col=1)
    return fig
