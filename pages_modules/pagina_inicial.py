"""Página inicial (landing page) exibida ao abrir o sistema."""
from pathlib import Path

import streamlit as st

from pages_modules.navegacao import PAGINA_ACADEMICA, PAGINA_SISTEMA, ir_para

ASSETS = Path(__file__).resolve().parent.parent / "assets"
VIDEO_APRESENTACAO = ASSETS / "video" / "apresentacao.mp4"
CSS_LANDING = ASSETS / "landing.css"

FAIXA_RESULTADOS = [
    "<b>79</b> eventos avaliados no paradigma event-level",
    "<b>YOLOv11s</b> com acurácia de 70,8% e F1 de 0,656",
    "Falsos positivos de <b>22 para 6</b> com o limitador de velocidade",
    "<b>100%</b> de precisão no UR Fall Detection Dataset",
    "Três bases: escola simulada, <b>UR Fall</b> e vídeos públicos",
    "Regra de queda: <b>AR &lt; 0,75</b> e <b>v ≥ 15 px/f</b>",
]

NUMEROS = [
    # (alvo, antes, depois, legenda)
    (79, "", "", "eventos contínuos avaliados (event-level)"),
    (656, "0,", "", "de F1-score no YOLOv11s, o melhor equilíbrio"),
    (6, "", "", "falsos positivos com velocidade, contra 22 sem ela"),
    (100, "", "%", "de precisão no UR Fall Detection Dataset"),
]

CARDS = [
    ("report", "#7b2d26", "O problema", "Quedas na escola",
     "Quedas respondem por quase 60% dos atendimentos de emergência de crianças em ambiente escolar, e a "
     "supervisão humana sofre com pontos cegos, grupos grandes e fadiga.",
     "≈ 60% dos atendimentos"),
    ("visibility", "#1f3a5f", "A abordagem", "Postura + velocidade",
     "O YOLOv11 detecta cada pessoa; a razão de aspecto da caixa indica a postura e a velocidade do "
     "centroide confirma se a chegada ao chão foi brusca.",
     "AR &lt; 0,75 e v ≥ 15 px/f"),
    ("monitoring", "#4f6d5a", "Os resultados", "Menos alarmes falsos",
     "O limitador cinemático derrubou os falsos positivos e o YOLOv11s mostrou o melhor equilíbrio entre "
     "acurácia e estabilidade da caixa.",
     "FP: 22 → 6"),
    ("view_in_ar", "#9a7b2f", "As limitações", "O eixo Z",
     "Quedas na direção da câmera geram pouco deslocamento em pixels. Normalização de escala e estimativa "
     "de pose são os próximos passos.",
     "Trabalhos futuros"),
]

ETAPAS_TUTORIAL = [
    ("Detecção da pessoa",
     "O YOLOv11 localiza cada pessoa no quadro e desenha a caixa delimitadora, com um ID que acompanha a "
     "pessoa entre os quadros."),
    ("Postura pela geometria",
     "A razão de aspecto <code>AR = h/w</code> indica a postura: em pé, sentado ou deitado "
     "(<code>AR &lt; 0,75</code>)."),
    ("Velocidade do centroide",
     "O deslocamento do centro da caixa entre quadros dá a velocidade em pixels por quadro, normalizada "
     "para 30 FPS."),
    ("Confirmação da queda",
     "O alerta só dispara com postura deitada <b>e</b> <code>v ≥ 15 px/f</code>: quem apenas se deita "
     "não gera alarme."),
]

PRIMEIROS_PASSOS = [
    ("Abra o sistema", "Clique em <b>Get Started</b>. A barra lateral lista os módulos e o ícone da casa "
                       "volta para esta página."),
    ("Conheça a pesquisa", "Em <b>Visão Geral e Artigo</b>, veja o resumo, a heurística bimodal e as bases "
                           "de dados."),
    ("Configure a análise", "No <b>Estúdio de Inferência</b>, escolha o modelo (YOLOv11s é o padrão) e o "
                            "modo de análise."),
    ("Envie um vídeo", "Faça o upload de um .mp4, .avi ou .mov, ou use uma amostra da pasta "
                       "<code>videos/</code>, e inicie."),
    ("Audite as ocorrências", "Em <b>Auditoria e Logs</b>, filtre os eventos e exporte em CSV, Excel ou "
                              "PDF."),
    ("Compare os modelos", "No <b>Dashboard Científico</b>, confira as tabelas e matrizes de confusão do "
                           "artigo."),
]


def _carregar_css():
    if CSS_LANDING.is_file():
        st.markdown(f"<style>{CSS_LANDING.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _botoes(prefixo: str, academico: bool = True):
    colunas = st.columns([1, 1.55] if academico else [1], gap="small")
    with colunas[0]:
        st.button("Get Started", type="primary", icon=":material/arrow_forward:", on_click=ir_para,
                  args=(PAGINA_SISTEMA,), width="stretch", key=f"{prefixo}_iniciar")
    if academico:
        with colunas[1]:
            st.button("Informações Acadêmicas", icon=":material/school:", on_click=ir_para,
                      args=(PAGINA_ACADEMICA,), width="stretch", key=f"{prefixo}_academico")


def _barra_superior():
    with st.container(key="lp_nav"):
        col_marca, col_botoes = st.columns([2.1, 1.9], vertical_alignment="center")
        with col_marca:
            st.markdown("""
            <div class="lp-marca">
                <div class="selo">DQ</div>
                <div>
                    <div class="nome">Sistema de Análise e Detecção de Quedas</div>
                    <div class="sub">UFPI · CEAD · Tecnologia em Gestão de Dados</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_botoes:
            _botoes("nav")


def _hero():
    col_texto, col_video = st.columns([1, 1.08], gap="large", vertical_alignment="center")
    with col_texto:
        st.markdown("""
        <div class="lp-entrada lp-d1"><span class="lp-rotulo"><span class="ponto"></span>
        Prova de conceito em visão computacional</span></div>
        <div class="lp-titulo lp-entrada lp-d2">Detecção de quedas em <em>tempo real</em> para ambientes
        escolares</div>
        <div class="lp-frase lp-entrada lp-d3">Cada segundo conta quando uma criança cai. Detectar a queda no
        instante em que ela acontece e ignorar tudo o que não é queda.</div>
        <div class="lp-descricao lp-entrada lp-d4">O sistema combina o detector YOLOv11 com uma heurística de
        postura e velocidade para separar a queda real de quem apenas se senta ou se deita, reduzindo os alarmes
        falsos que inviabilizam a supervisão automatizada.</div>
        """, unsafe_allow_html=True)
        _botoes("hero")
        st.markdown("""
        <div class="lp-selos lp-entrada lp-d4"><span>YOLOv11</span><span>Event-level evaluation</span>
        <span>TCC · UFPI</span></div>
        """, unsafe_allow_html=True)
    with col_video:
        with st.container(key="lp_monitor"):
            st.markdown("""
            <div class="lp-monitor-topo"><span class="lp-ao-vivo">Demonstração</span>
            <span>CAM 01 · Sala 03</span></div>
            """, unsafe_allow_html=True)
            if VIDEO_APRESENTACAO.is_file():
                st.video(str(VIDEO_APRESENTACAO), autoplay=True, loop=True, muted=True)
            st.markdown("""
            <div class="lp-monitor-base"><span>Cena 1: queda brusca → alerta</span>
            <span>Cena 2: deitar devagar → sem alerta</span></div>
            """, unsafe_allow_html=True)


def _faixa():
    itens = "".join(f"<span>{item}</span>" for item in FAIXA_RESULTADOS * 2)
    st.markdown(f'<div class="lp-faixa"><div class="lp-faixa-trilho">{itens}</div></div>',
                unsafe_allow_html=True)


def _numeros():
    blocos = "".join(
        f'<div><div class="lp-num" style="--alvo: {alvo}" data-antes="{antes}" data-depois="{depois}"></div>'
        f'<div class="lp-num-legenda">{legenda}</div></div>'
        for alvo, antes, depois, legenda in NUMEROS
    )
    st.markdown(f'<div class="lp-numeros lp-entrada lp-d3">{blocos}</div>', unsafe_allow_html=True)


def _cabecalho_secao(numero: str, titulo: str, texto: str):
    st.markdown(f"""
    <div class="lp-secao lp-revelar">
        <div><div class="num">{numero}</div><h2>{titulo}</h2></div>
        <p>{texto}</p>
    </div>
    """, unsafe_allow_html=True)


def _cards():
    _cabecalho_secao("01 · Visão geral", "A pesquisa em quatro pontos",
                     "Do problema que motivou o trabalho às limitações que orientam os próximos passos.")
    cards = "".join(f"""
        <div class="lp-card lp-revelar" style="--cor: {cor}">
            <div class="lp-icone">{icone}</div>
            <div class="rotulo">{rotulo}</div>
            <h3>{titulo}</h3>
            <p>{texto}</p>
            <div class="dado">{dado}</div>
        </div>""" for icone, cor, rotulo, titulo, texto, dado in CARDS)
    st.markdown(f'<div class="lp-cards">{cards}</div>', unsafe_allow_html=True)


def _tutorial():
    _cabecalho_secao("02 · Tutorial breve", "Como a detecção funciona",
                     "Quatro etapas executadas a cada quadro do vídeo. A etapa em destaque avança sozinha.")
    etapas = "".join(f"""
        <div class="lp-etapa">
            <div class="lp-no">{i}</div>
            <div class="lp-etapa-corpo"><h4>{titulo}</h4><p>{texto}</p></div>
        </div>""" for i, (titulo, texto) in enumerate(ETAPAS_TUTORIAL, start=1))
    st.markdown(f'<div class="lp-tutorial lp-revelar"><div class="lp-trilho"></div>'
                f'<div class="lp-etapas">{etapas}</div></div>', unsafe_allow_html=True)


def _primeiros_passos():
    _cabecalho_secao("03 · Primeiros passos", "Do primeiro clique ao relatório",
                     "Um roteiro curto para explorar o sistema pela primeira vez.")
    passos = "".join(f"""
        <div class="lp-passo lp-revelar">
            <div class="n">{i:02d}</div>
            <div><h4>{titulo}</h4><p>{texto}</p></div>
        </div>""" for i, (titulo, texto) in enumerate(PRIMEIROS_PASSOS, start=1))
    st.markdown(f'<div class="lp-passos">{passos}</div>', unsafe_allow_html=True)


def _chamada_final():
    with st.container(key="lp_cta"):
        col_texto, col_botoes = st.columns([1.45, 1.25], vertical_alignment="center")
        with col_texto:
            st.markdown("""
            <div class="lp-cta-titulo">Pronto para analisar um vídeo?</div>
            <div class="lp-cta-texto">Abra o Estúdio de Inferência, escolha o modelo e veja a heurística
            funcionando quadro a quadro.</div>
            """, unsafe_allow_html=True)
        with col_botoes:
            _botoes("cta")


def render_pagina_inicial():
    _carregar_css()
    _barra_superior()
    _hero()
    _faixa()
    _numeros()
    _cards()
    _tutorial()
    _primeiros_passos()
    _chamada_final()
    st.markdown("""
    <div class="lp-rodape">
        <span>Sistema de Análise e Detecção de Quedas · Trabalho de Conclusão de Curso · CSTGD · CEAD · UFPI</span>
        <span>Informações sobre autoria e orientação na seção Informações Acadêmicas</span>
    </div>
    """, unsafe_allow_html=True)
