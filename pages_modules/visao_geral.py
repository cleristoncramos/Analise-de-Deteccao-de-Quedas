import streamlit as st


def _divisor():
    """Linha fina e discreta entre seções (um div, para não herdar o estilo forte do <hr>)."""
    st.markdown('<div class="divisor-fino"></div>', unsafe_allow_html=True)

def render_visao_geral():
    st.title("Sobre a pesquisa")
    st.caption("Prova de Conceito (PoC) para Prevenção de Acidentes em Ambientes Escolares baseada em Visão Computacional")

    # Header de Autoria e Instituição
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Universidade Federal do Piauí (UFPI)**  
        *Centro de Educação Aberta e a Distância (CEAD)*  
        *Curso Superior de Tecnologia em Gestão de Dados (CSTGD)*  
        
        **Autor**: Brendo Alves Soares  
        **Orientadores**: Wener Borges de Sampaio e Martony Demes da Silva  
        """)
    with col2:
        st.caption("Os resultados completos da avaliação (tabelas por base de dados e por modelo) estão no artigo e "
                   "no README do repositório. No sistema, a tela Avaliação do conjunto recalcula as métricas nos "
                   "70 vídeos do UR Fall.")

    _divisor()

    # Seções em abas organizadas
    tab_resumo, tab_metodologia, tab_dados = st.tabs([
        "Resumo do Trabalho",
        "Heurística Bimodal (AR + Velocidade)",
        "Base de Dados Heterogênea",
    ])

    with tab_resumo:
        st.subheader("Contexto e Motivação")
        st.markdown("""
        O ambiente escolar registra anualmente elevados índices de internações infantis decorrentes de acidentes, 
        sendo as **quedas responsáveis por quase 60% dos atendimentos de emergência**. 
        
        A supervisão humana convencional enfrenta limitações severas devido a pontos cegos, múltiplos alunos e fadiga. 
        Este trabalho implementa uma **camada de monitoramento preditivo e ativo** baseada no **YOLOv11**, 
        focando estritamente na detecção da classe *pessoa* e na aplicação de heurísticas cinemáticas.
        """)
        _divisor()
        st.subheader("O sistema")
        st.markdown("""
        - **Estúdio de Inferência:** analisa vídeos do UR Fall, arquivos próprios, links do YouTube, transmissões
          ao vivo e a câmera do navegador, com o vídeo anotado, o gráfico de postura e velocidade e o gabarito.
        - **Avaliação do conjunto:** percorre os 70 vídeos do UR Fall e recalcula a matriz de confusão e as
          métricas a cada vídeo, ao lado da Tabela 2 do artigo.
        - **Auditoria e Logs:** reúne as ocorrências da sessão, com filtros e exportação em CSV, Excel e PDF.
        """)

    with tab_metodologia:
        st.subheader("Como Funciona a Heurística Bimodal?")
        st.markdown("""
        Diferente de abordagens que dependem apenas da geometria estática (que geram alarmes falsos quando uma pessoa senta ou deita deliberadamente), 
        o sistema condiciona o disparo do alerta à interseção lógica de **dois ramos analíticos simultâneos**:
        """)

        col_heur1, col_heur2 = st.columns(2)
        with col_heur1:
            st.markdown("""
            #### 1. Geometria Postural (Razão de Aspecto - AR)
            Calcula a proporção entre altura ($h$) e largura ($w$) da *bounding box*:
            $$AR = \\frac{h}{w}$$
            - **Deitado**: $AR < 0.75$ *(Postura de risco em potencial)*
            - **Sentado / Agachado**: $0.75 \\le AR \\le 1.20$
            - **Em pé**: $AR > 1.20$
            """)

        with col_heur2:
            st.markdown("""
            #### 2. Cinemática Escalar (Velocidade do Centroide - v)
            Calcula o deslocamento euclidiano do centroide $C(x,y)$ entre frames consecutivos para o mesmo ID:
            $$v = \\sqrt{(C_{x,t} - C_{x,t-1})^2 + (C_{y,t} - C_{y,t-1})^2}$$
            - **Mexendo-se pouco**: $v < 1.0\\text{ px/f}$
            - **Andando**: $1.0 \\le v < 30.0\\text{ px/f}$
            - **Correndo**: $v \\ge 30.0\\text{ px/f}$

            A velocidade é normalizada para a referência de 30 FPS, para que o limiar valha em vídeos de 25,
            30 ou 60 FPS.
            """)

        _divisor()
        st.success("""
        **Regra Lógica de Confirmação de Queda:**
        $$\\text{Queda} = \\text{Verdadeiro} \\iff (\\text{Postura} = \\text{"Deitado"}) \\land (v \\ge 15.0\\text{ px/frame})$$
        """)

    with tab_dados:
        st.subheader("Conjunto de Dados Heterogêneo")
        st.markdown("""
        Para contornar restrições legais e éticas referentes a imagens de menores (**LGPD** e **ECA**), 
        a validação utilizou três fontes complementares:
        """)
        _divisor()
        
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.markdown("""
            **1. Dataset Simulado (Escola)**
            - Gravado em sala de aula real com câmeras simulando monitoramento.
            - Movimentações em grupo, oclusões, ações voluntárias e quedas ensaiadas.
            """)
        with col_d2:
            st.markdown("""
            **2. UR Fall Detection Dataset**
            - Benchmark científico consolidado da literatura.
            - 30 quedas e 30 Atividades de Vida Diária (ADL) capturadas por Kinect.
            - No sistema, os 70 vídeos da câmera 0 (30 quedas e 40 ADL), usando só a imagem colorida.
            """)
        with col_d3:
            st.markdown("""
            **3. Vídeos de Domínio Público**
            - Cenas do mundo real extraídas da internet.
            - Teste de estresse com ruído, iluminação variável e perspectiva em profundidade (**Eixo Z**).
            """)
