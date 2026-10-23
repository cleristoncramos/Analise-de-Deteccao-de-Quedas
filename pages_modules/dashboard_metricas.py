import streamlit as st
import pandas as pd
import json
import os
from components.chart_builder import (
    criar_matriz_confusao,
    criar_grafico_comparativo_metricas,
    criar_grafico_radar_metricas,
    criar_grafico_reducao_falsos_positivos
)

@st.cache_data
def carregar_dados_benchmark():
    caminho_json = os.path.join("data", "tcc_benchmark_data.json")
    with open(caminho_json, "r", encoding="utf-8") as f:
        return json.load(f)

def render_dashboard_metricas():
    st.title("📊 Dashboard Científico e Métricas do Artigo")
    st.caption("Resultados empíricos consolidados da Prova de Conceito sob o paradigma Event-Level Evaluation (SBC)")

    dados = carregar_dados_benchmark()

    for aviso in dados["metadados"].get("pendencias", []):
        st.warning(f"**Pendência nos dados do artigo:** {aviso}")

    # Destaque das Métricas Globais do Melhor Modelo (maior F1 no cenário Com Velocidade)
    df_tabela4 = pd.DataFrame(dados["tabela_4_geral"])
    df_com = df_tabela4[df_tabela4["cenario"] == "Com Velocidade"]
    melhor = df_com.loc[df_com["f1_score"].idxmax()]
    base = df_tabela4[df_tabela4["cenario"] == "Sem Velocidade"].iloc[0]

    def delta_pp(metrica: str) -> str:
        return f"{(melhor[metrica] - base[metrica]) * 100:+.1f} p.p. vs Sem Vel"

    st.markdown(f"### 🏆 Performance do Modelo Recomendado: {melhor['modelo']} (Com Velocidade)")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Acurácia Global", f"{melhor['acuracia']:.1%}", delta_pp("acuracia"))
    with col2:
        st.metric("Precisão (P)", f"{melhor['precisao']:.1%}", delta_pp("precisao"))
    with col3:
        st.metric("Revocação (Recall)", f"{melhor['recall']:.1%}", delta_pp("recall"))
    with col4:
        st.metric("F1-Score Global", f"{melhor['f1_score']:.3f}", "Maior entre os modelos Com Velocidade")

    st.divider()

    aba_geral, aba_simulado, aba_ur, aba_publico, aba_analise = st.tabs([
        "🌐 Comparativo Geral (79 Eventos)",
        "🏫 Dataset Simulado (Escola)",
        "🔬 UR Fall Dataset",
        "🌍 Vídeos de Domínio Público",
        "💡 Análise Crítica dos Modelos"
    ])

    with aba_geral:
        st.subheader("Tabela 4: Comparativo Consolidado de Todos os Cenários")
        df_geral = df_tabela4
        st.dataframe(
            df_geral.style.highlight_max(subset=["acuracia", "precisao", "f1_score"], color="rgba(34, 197, 94, 0.25)")
                          .highlight_min(subset=["fp"], color="rgba(56, 189, 248, 0.25)"),
            width="stretch"
        )

        col_g1, col_g2 = st.columns([3, 2])
        with col_g1:
            fig_metricas = criar_grafico_comparativo_metricas(df_geral, "Comparativo Geral de Métricas por Modelo")
            st.plotly_chart(fig_metricas)
        with col_g2:
            fig_fp = criar_grafico_reducao_falsos_positivos(df_geral)
            st.plotly_chart(fig_fp)

        st.subheader("Explorador de Matriz de Confusão por Modelo")
        opcoes_modelos = df_geral["modelo"].tolist()
        modelo_sel = st.selectbox("Selecione o modelo para inspecionar:", opcoes_modelos, index=1)
        linha_sel = df_geral[df_geral["modelo"] == modelo_sel].iloc[0]

        col_mat, col_rad = st.columns(2)
        with col_mat:
            fig_matriz = criar_matriz_confusao(
                tn=int(linha_sel["tn"]),
                fp=int(linha_sel["fp"]),
                fn=int(linha_sel["fn"]),
                tp=int(linha_sel["tp"]),
                titulo=f"Matriz de Confusão: {modelo_sel}"
            )
            st.plotly_chart(fig_matriz)
        with col_rad:
            fig_radar = criar_grafico_radar_metricas(linha_sel.to_dict(), f"Perfil de Métricas: {modelo_sel}")
            st.plotly_chart(fig_radar)

    with aba_simulado:
        st.subheader("Tabela 1: Métricas de Detecção no Dataset Simulado (Escola)")
        st.markdown("""
        Ambiente com dinâmica de pátio/sala de aula com oclusões, pessoas em grupo e ações voluntárias (ex: agachar para pegar objeto).
        """)
        df_sim = pd.DataFrame(dados["tabela_1_simulado"])
        st.dataframe(df_sim, width="stretch")
        fig_sim = criar_grafico_comparativo_metricas(df_sim, "Métricas no Cenário Simulado")
        st.plotly_chart(fig_sim)

    with aba_ur:
        st.subheader("Tabela 2: Métricas de Detecção no UR Fall Detection Dataset")
        st.markdown("""
        Benchmark científico internacional: 60 eventos avaliados (30 quedas e 30 atividades de vida diária) capturados por Kinect.
        **Destaque**: A inclusão da heurística cinemática atingiu **100% de Precisão (zero falsos positivos)** em todas as variantes YOLOv11!
        """)
        df_ur = pd.DataFrame(dados["tabela_2_ur_fall"])
        st.dataframe(df_ur, width="stretch")
        fig_ur = criar_grafico_comparativo_metricas(df_ur, "Métricas no UR Fall Dataset")
        st.plotly_chart(fig_ur)

    with aba_publico:
        st.subheader("Tabela 3: Vídeos de Domínio Público (Teste de Estresse do Eixo Z)")
        st.markdown("""
        Cenas reais da internet gravadas em planos abertos (*long shot*) onde a movimentação ocorre predominantemente em profundidade.
        """)
        df_pub = pd.DataFrame(dados["tabela_3_publico"])
        st.dataframe(df_pub, width="stretch")
        st.warning("""
        **Vulnerabilidade Mapeada**: No modelo Com Velocidade, o deslocamento em pixels do centroide ficou abaixo de $15.0\\text{ px/f}$ devido à distância da câmera. 
        Isso comprova que a visão computacional enxergou a ação, mas a heurística matemática bidimensional a bloqueou pela falta de normalização de escala em profundidade.
        """)

    with aba_analise:
        st.subheader("Análise Qualitativa dos Resultados Científicos")
        st.markdown("""
        #### 1. Por que o YOLOv11x teve pior desempenho que o YOLOv11s?
        Em redes convolucionais profundas como o *YOLOv11x*, sob situações de oclusão frequentes no ambiente escolar, 
        a caixa delimitadora tenta se ajustar aos detalhes anatômicos mais finos a cada fração de segundo. 
        Essa micro-oscilação (**spatial jitter**) altera artificialmente o centroide $C(x,y)$, gerando velocidades espúrias que disparam falsos alarmes de queda. 
        Modelos mais compactos e generalistas (*YOLOv11s* e *v11m*) mantêm uma delimitação mais estável, garantindo a eficácia da filtragem cinemática.

        #### 2. O Trade-Off entre Precisão e Recall
        - **Sem Velocidade**: Alto recall (detecta praticamente todas as quedas), porém com **fadiga de alarme** inaceitável para uso real (22 falsos positivos gerais).
        - **Com Velocidade**: Elimina os alarmes falsos (Precisão sobe de 54% para até 100%), protegendo as equipes escolares de alertas repetidos sem motivo.
        """)
