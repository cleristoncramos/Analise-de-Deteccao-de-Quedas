import streamlit as st

def render_visao_geral():
    st.title("Sistema de Análise e Detecção de Quedas")
    st.caption("Prova de Conceito (PoC) para Prevenção de Acidentes em Ambientes Escolares baseada em Visão Computacional")

    # Header de Autoria e Instituição
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        **Universidade Federal do Piauí (UFPI)**  
        *Centro de Educação Aberta e a Distância (CEAD)*  
        *Curso Superior de Tecnologia em Gestão de Dados (CSTGD)*  
        
        **Autor**: Brendo A. Soares  
        **Orientadores**: Wener B. de Sampaio e Martony D. da Silva  
        **Paradigma de Avaliação**: *Event-Level Evaluation* (79 eventos contínuos)
        """)
    with col2:
        st.info("**Destaque Científico**\n\nO modelo **YOLOv11s** demonstrou ser o ponto de equilíbrio ideal (*sweet spot*) entre acurácia e estabilidade de centroide.")

    st.divider()

    # Seções em abas organizadas
    tab_resumo, tab_metodologia, tab_dados, tab_conclusoes = st.tabs([
        "Resumo do Trabalho",
        "Heurística Bimodal (AR + Velocidade)",
        "Base de Dados Heterogênea",
        "Principais Conclusões"
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

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric(label="Eventos Avaliados", value="79 eventos", help="Total de eventos contínuos analisados sob o paradigma Event-Level.")
        with col_b:
            st.metric(label="Melhor Modelo", value="YOLOv11s", help="Maior F1-score global (0.656) e Acurácia (70.89%).")
        with col_c:
            st.metric(label="Precisão UR Fall", value="100%", help="Zero falsos positivos com limitador cinemático ativado.")

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
            """)

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
            """)
        with col_d3:
            st.markdown("""
            **3. Vídeos de Domínio Público**
            - Cenas do mundo real extraídas da internet.
            - Teste de estresse com ruído, iluminação variável e perspectiva em profundidade (**Eixo Z**).
            """)

    with tab_conclusoes:
        st.subheader("Descobertas Fundamentais do Estudo")
        st.markdown("""
        1. **Eliminação da Fadiga de Alarme**:
           - A heurística puramente postural (*Sem Velocidade*) gerou **22 falsos positivos**, tornando o sistema inviável na prática.
           - A integração da velocidade escalar reduziu os falsos positivos para **6 no modelo YOLOv11m** e atingiu **100% de precisão no UR Fall**.
        2. **O Fenômeno do Overfitting Espacial**:
           - Modelos excessivamente complexos (*Large* e *Extra Large*) apresentaram oscilação contínua de *bounding box* sob oclusão, criando falsos vetores de velocidade que degradaram a precisão.
           - O modelo **YOLOv11s** provou ser a arquitetura mais equilibrada para aplicação real.
        3. **O Gargalo do Eixo Z e Trabalhos Futuros**:
           - Quedas na linha de visão direta da câmera geram baixo deslocamento $2D$ em pixels, sugerindo a necessidade de normalização de escala e estimativa de pose tridimensional (*Pose Estimation*).
        """)
