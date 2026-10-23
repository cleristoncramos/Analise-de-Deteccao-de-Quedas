import streamlit as st
import numpy as np
import plotly.graph_objects as go

from core.heuristics import calcular_velocidade_normalizada

def render_laboratorio_experimental():
    st.title("Laboratório Experimental: Trabalhos Futuros")
    st.caption("Investigação e prototipagem de soluções para as limitações identificadas no TCC (Eixo Z e Pose Estimation)")

    tab_eixo_z, tab_pose = st.tabs([
        "Solução 1: Normalização de Escala (Mitigação do Eixo Z)",
        "Solução 2: Transição para Pose Estimation (YOLO11-Pose)"
    ])

    with tab_eixo_z:
        st.subheader("1. O Problema da Distância Focal e o Eixo Z")
        st.markdown("""
        No artigo científico, observou-se que em planos abertos (*long shot*) a velocidade calculada em **pixels absolutos** 
        sofre forte atenuação pela distância da câmera:
        - Uma queda a **2 metros** da câmera pode deslocar o centroide em **40 pixels/frame**.
        - A **mesma queda a 10 metros** da câmera desloca o centroide em apenas **6 pixels/frame**, ficando abaixo do limiar ($v < 15\\text{ px/f}$) e gerando um Falso Negativo.
        """)

        st.info("""
        **Fórmula Proposta para Normalização Cinemática:**
        $$v_{norm} = \\left( \\frac{v_{pixel}}{\\sqrt{w \\times h}} \\right) \\times 100$$
        Onde $w$ e $h$ são as dimensões da bounding box da pessoa.
        """)

        st.subheader("Simulador Interativo de Escala")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            distancia_simulada = st.select_slider(
                "Distância do Indivíduo até a Câmera",
                options=["Muito Próximo (2m)", "Médio (5m)", "Afastado (10m)", "Muito Afastado (20m)"],
                value="Afastado (10m)"
            )
            mapa_alturas = {
                "Muito Próximo (2m)": 450,
                "Médio (5m)": 240,
                "Afastado (10m)": 100,
                "Muito Afastado (20m)": 45
            }
            altura_bbox = mapa_alturas[distancia_simulada]
            largura_bbox = int(altura_bbox * 0.45)
            st.metric("Dimensões Estimadas da BBox", f"{largura_bbox}px x {altura_bbox}px")

        with col_s2:
            velocidade_bruta = st.slider("Velocidade Bruta em Pixels (v)", 1.0, 50.0, 10.0, 0.5)
            
            # Cálculo normalizado (mesma fórmula do pipeline em core/heuristics.py)
            v_norm = calcular_velocidade_normalizada(velocidade_bruta, largura_bbox, altura_bbox)

            st.metric("Velocidade Normalizada (v_norm)", f"{v_norm:.2f}% / f")

        # Comparativo de decisão
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            queda_bruta = velocidade_bruta >= 15.0
            if queda_bruta:
                st.success("Heurística Absoluta Original: QUEDA DETECTADA ($v \\ge 15.0$)")
            else:
                st.error("Heurística Absoluta Original: FALHA ($v < 15.0$) - Falso Negativo!")

        with col_res2:
            limiar_norm = 12.0  # Limiar calibrado percentual
            queda_norm = v_norm >= limiar_norm
            if queda_norm:
                st.success("Heurística Normalizada Proposta: QUEDA DETECTADA ($v_{norm} \\ge 12.0\\%$)")
            else:
                st.warning("Ação voluntária controlada")

        # Gráfico de sensibilidade
        alturas_range = np.linspace(30, 500, 100)
        v_pixels_fixa = 12.0
        v_norms = [calcular_velocidade_normalizada(v_pixels_fixa, h * 0.45, h) for h in alturas_range]

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=alturas_range,
            y=v_norms,
            mode="lines",
            line=dict(color="#1f3a5f", width=3),
            name="Velocidade Normalizada (%)"
        ))
        fig_sens.add_hline(y=12.0, line_dash="dash", line_color="#7b2d26", annotation_text="Limiar de Queda Normalizado (12%)")
        fig_sens.update_layout(
            title="Sensibilidade da Velocidade Normalizada conforme a Escala (Altura da BBox)",
            xaxis_title="Altura da Bounding Box (pixels)",
            yaxis_title="Velocidade Normalizada (% / frame)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1d1d1b"),
            height=320,
            margin=dict(l=30, r=30, t=50, b=30)
        )
        st.plotly_chart(fig_sens)

    with tab_pose:
        st.subheader("2. Transição para Pose Estimation (YOLO11-Pose)")
        st.markdown("""
        Conforme apontado na seção de **Trabalhos Futuros** do artigo, a evolução para **Pontos-Chave Anatômicos (17 Keypoints)** 
        permite estimar a orientação do vetor coluna vertebral no espaço tridimensional, eliminando a dependência de aspecto da caixa:
        """)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.markdown("""
            #### Abordagem Atual: Bounding Box
            - Depende da proporção $h/w$.
            - Vulnerável quando o aluno cai na direção da câmera (Eixo Z).
            - Baixo custo computacional (roda facilmente em CPU escolar).
            """)
        with col_p2:
            st.markdown("""
            #### Abordagem Futura: YOLO11-Pose
            - Rastreamento da inclinação angular do tronco:
              $$\\theta = \\arctan\\left(\\frac{y_{ombro} - y_{quadril}}{x_{ombro} - x_{quadril}}\\right)$$
            - Imune a oclusões parciais da caixa.
            - Requer maior capacidade computacional ou aceleração gráfica.
            """)
