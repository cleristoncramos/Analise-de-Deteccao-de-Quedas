"""Página inicial (landing page) exibida ao abrir o sistema."""
from pathlib import Path

import streamlit as st

from pages_modules.navegacao import PAGINA_ACADEMICA, PAGINA_SISTEMA, ir_para

ASSETS = Path(__file__).resolve().parent.parent / "assets"
VIDEO_APRESENTACAO = ASSETS / "video" / "apresentacao.mp4"
ANIMACAO_DEMO = ASSETS / "demo_heuristica.gif"

ETAPAS_TUTORIAL = [
    ("Etapa 1", "Detecção da pessoa",
     "O YOLOv11 localiza cada pessoa no quadro e desenha a caixa delimitadora (bounding box)."),
    ("Etapa 2", "Postura pela geometria",
     "A razão de aspecto AR = h/w da caixa indica a postura: em pé, sentado ou deitado (AR &lt; 0,75)."),
    ("Etapa 3", "Velocidade do centroide",
     "O rastreador acompanha o centro da caixa entre quadros e mede o deslocamento em pixels por quadro."),
    ("Etapa 4", "Confirmação da queda",
     "O alerta só dispara quando a postura é deitada e a velocidade passa de 15 px/f, o que filtra quem apenas se deita."),
]

PRIMEIROS_PASSOS = [
    "Clique em <b>Get Started</b> para abrir o sistema. A barra lateral lista os módulos disponíveis.",
    "Em <b>Visão Geral e Artigo</b>, leia o resumo da pesquisa e a heurística bimodal de postura e velocidade.",
    "Abra o <b>Estúdio de Inferência</b>, escolha o modelo (o YOLOv11s é o padrão recomendado) e o modo de análise.",
    "Envie um vídeo (.mp4, .avi ou .mov) ou use uma amostra da pasta <code>videos/</code> e inicie a inferência.",
    "Confira as ocorrências em <b>Auditoria e Logs</b> e exporte o relatório em CSV, Excel ou PDF.",
    "Compare os resultados do artigo no <b>Dashboard Científico</b>. O ícone da casa na barra lateral volta para esta página.",
]


def _cabecalho():
    col_faixa, col_botao = st.columns([3, 1], vertical_alignment="center")
    with col_faixa:
        st.markdown(
            '<div class="faixa-institucional"><span>Universidade Federal do Piauí · CEAD · '
            'Tecnologia em Gestão de Dados</span><span>Trabalho de Conclusão de Curso</span></div>',
            unsafe_allow_html=True,
        )
    with col_botao:
        st.button("Informações Acadêmicas", icon=":material/school:", on_click=ir_para,
                  args=(PAGINA_ACADEMICA,), width="content", key="topo_academico")


def _hero():
    st.markdown("""
    <div class="hero">
        <div class="rotulo">Prova de conceito em visão computacional</div>
        <div class="nome">Sistema de Análise e Detecção de Quedas</div>
        <div class="frase">Cada segundo conta quando uma criança cai. Detectar a queda no instante em que ela
        acontece e ignorar tudo o que não é queda.</div>
        <div class="descricao">Monitoramento de vídeo para ambientes escolares que combina o detector YOLOv11
        com uma heurística de postura e velocidade, reduzindo os alarmes falsos que tornam inviável a
        supervisão automatizada.</div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col_a, col_b, _ = st.columns([1, 1.25, 2.75])
    with col_a:
        st.button("Get Started", type="primary", icon=":material/arrow_forward:", on_click=ir_para,
                  args=(PAGINA_SISTEMA,), width="stretch", key="get_started")
    with col_b:
        st.button("Informações Acadêmicas", icon=":material/school:", on_click=ir_para,
                  args=(PAGINA_ACADEMICA,), width="content", key="hero_academico")


def _numeros():
    st.markdown("""
    <div class="numeros">
        <div><div class="valor">79</div><div class="legenda">eventos avaliados (event-level)</div></div>
        <div><div class="valor">YOLOv11s</div><div class="legenda">melhor equilíbrio entre acurácia e estabilidade</div></div>
        <div><div class="valor">22 → 6</div><div class="legenda">falsos positivos sem e com o limitador de velocidade</div></div>
        <div><div class="valor">100%</div><div class="legenda">de precisão no UR Fall Detection Dataset</div></div>
    </div>
    """, unsafe_allow_html=True)


def _secao(numero: str, titulo: str):
    st.markdown(f'<div class="secao-titulo"><span class="numero">{numero}</span>{titulo}</div>',
                unsafe_allow_html=True)


def _video():
    _secao("01", "Demonstração em vídeo")
    col_video, col_texto = st.columns([3, 2], gap="large")
    with col_video:
        if VIDEO_APRESENTACAO.is_file():
            st.video(str(VIDEO_APRESENTACAO))
            legenda = "Vídeo de apresentação do sistema."
        else:
            st.image(str(ANIMACAO_DEMO), width="stretch")
            legenda = ("Figura animada: na cena 1 a queda brusca dispara o alerta; na cena 2 a pessoa se deita "
                       "devagar e nenhum alerta é emitido.")
        st.markdown(f'<div class="legenda-figura">{legenda}</div>', unsafe_allow_html=True)
    with col_texto:
        st.markdown("""
        **O que o vídeo mostra**

        A mesma postura deitada aparece nas duas cenas. A diferença está na velocidade com que o corpo chega ao chão:

        - **Queda brusca:** razão de aspecto abaixo de 0,75 e velocidade acima de 15 px/f, o que gera alerta.
        - **Deitar-se devagar:** a postura também é deitada, mas a velocidade fica baixa e o sistema não alerta.

        Esse é o mecanismo que reduziu os falsos positivos de 22 para 6 no experimento do artigo.
        """)


def _tutorial():
    _secao("02", "Tutorial breve: como a detecção funciona")
    colunas = st.columns(len(ETAPAS_TUTORIAL))
    for coluna, (etapa, titulo, texto) in zip(colunas, ETAPAS_TUTORIAL):
        with coluna:
            st.markdown(f"""
            <div class="cartao">
                <div class="etapa">{etapa}</div>
                <div class="titulo">{titulo}</div>
                <div class="texto">{texto}</div>
            </div>
            """, unsafe_allow_html=True)


def _primeiros_passos():
    _secao("03", "Primeiros passos")
    itens = "".join(f"<li>{passo}</li>" for passo in PRIMEIROS_PASSOS)
    st.markdown(f'<ol class="passos">{itens}</ol>', unsafe_allow_html=True)
    st.write("")
    col_a, _ = st.columns([1, 4])
    with col_a:
        st.button("Get Started", type="primary", icon=":material/arrow_forward:", on_click=ir_para,
                  args=(PAGINA_SISTEMA,), width="stretch", key="get_started_rodape")


def render_pagina_inicial():
    _cabecalho()
    _hero()
    st.write("")
    _numeros()
    st.divider()
    _video()
    st.divider()
    _tutorial()
    st.divider()
    _primeiros_passos()
    st.markdown("""
    <div class="rodape">
        <span>Sistema de Análise e Detecção de Quedas · TCC · CSTGD · CEAD · UFPI</span>
        <span>Autor: Brendo A. Soares · Orientadores: Wener B. de Sampaio e Martony D. da Silva</span>
    </div>
    """, unsafe_allow_html=True)
