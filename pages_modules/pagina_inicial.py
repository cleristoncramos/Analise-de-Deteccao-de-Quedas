"""Página inicial (landing page) exibida ao abrir o sistema."""
from pathlib import Path

import streamlit as st

from pages_modules.navegacao import PAGINA_ACADEMICA, abrir_estudio_com_exemplo, ir_para

ASSETS = Path(__file__).resolve().parent.parent / "assets"
VIDEO_APRESENTACAO = ASSETS / "video" / "apresentacao.mp4"
CSS_LANDING = ASSETS / "landing.css"

FAIXA_RESULTADOS = [
    "Analise <b>vídeos próprios</b>, amostras do <b>UR Fall</b>, links do <b>YouTube</b> e transmissões ao vivo",
    "Câmera do <b>navegador</b>: transforme o computador em um detector",
    "Compare lado a lado a regra <b>sem</b> e <b>com velocidade</b>",
    "Gráfico de <b>postura e velocidade</b> quadro a quadro",
    "Vídeo anotado em <b>velocidade real</b> e relatório em CSV, Excel ou PDF",
    "Regra de queda: <b>AR &lt; 0,75</b> e <b>v ≥ 15 px/f</b>",
]

NUMEROS = [
    # (alvo, antes, depois, ícone, cor, rótulo, legenda)
    (5, "", "", "input", "#1f3a5f", "Fontes de vídeo",
     "Amostra, upload, link ou transmissão, câmera do navegador e webcam local."),
    (5, "", "", "neurology", "#7b2d26", "Modelos YOLOv11",
     "Do Nano ao Extra Large; o Small é o padrão recomendado."),
    (8, "", "", "video_library", "#9a7b2f", "Amostras com gabarito",
     "Vídeos do UR Fall prontos para testar, com o resultado esperado."),
    (2, "", "", "compare", "#4f6d5a", "Modos comparados",
     "Regra estática e regra com velocidade, lado a lado no mesmo vídeo."),
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
    ("compare", "#4f6d5a", "A ferramenta", "Veja a diferença",
     "No modo A/B, o mesmo vídeo é analisado com e sem o limitador de velocidade: quem apenas se deita não "
     "gera alarme na proposta.",
     "Lado a lado"),
    ("fact_check", "#9a7b2f", "Transparência", "O sistema pode errar",
     "Como todo detector, há falhas, sobretudo em quedas na direção da câmera. As amostras trazem o gabarito "
     "para conferir cada resultado.",
     "Gabarito em cada amostra"),
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
    ("Abra o Estúdio", "Clique em <b>Get Started</b>: o Estúdio abre com uma amostra do UR Fall pronta, no "
                       "modo lado a lado."),
    ("Rode o exemplo", "Clique em <b>Iniciar Análise</b> e acompanhe a prévia, o gráfico de postura e "
                       "velocidade e o gabarito."),
    ("Teste outros vídeos", "Escolha outra amostra, envie um vídeo seu ou cole um link do YouTube ou de uma "
                            "transmissão."),
    ("Use a câmera", "Em <b>Câmera do Navegador</b>, permita o acesso e veja a detecção ao vivo."),
    ("Ajuste a análise", "Na barra lateral, troque o modelo YOLOv11 e o modo, ou calibre os limiares."),
    ("Audite as ocorrências", "Em <b>Auditoria e Logs</b>, filtre os eventos e exporte em CSV, Excel ou "
                              "PDF."),
]


def _carregar_css():
    if CSS_LANDING.is_file():
        st.markdown(f"<style>{CSS_LANDING.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _botoes(prefixo: str, academico: bool = True):
    colunas = st.columns([1, 1.55] if academico else [1], gap="small")
    with colunas[0]:
        st.button("Get Started", type="primary", icon=":material/arrow_forward:", on_click=abrir_estudio_com_exemplo,
                  width="stretch", key=f"{prefixo}_iniciar")
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
                    <div class="sub">Visão computacional aplicada à segurança escolar</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_botoes:
            _botoes("nav")


def _hero():
    col_texto, col_video = st.columns([1, 1.08], gap="large", vertical_alignment="center")
    with col_texto:
        st.markdown("""
        <div class="lp-instituicao lp-entrada lp-d1">Universidade Federal do Piauí · CEAD · CSTGD</div>
        <div class="lp-entrada lp-d1"><span class="lp-rotulo"><span class="ponto"></span>
        TCC · Prova de conceito</span></div>
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
    blocos = "".join(f"""
        <div class="lp-indicador" style="--cor: {cor}">
            <div class="lp-indicador-topo"><span class="lp-indicador-rotulo">{rotulo}</span>
            <span class="lp-indicador-icone">{icone}</span></div>
            <div class="lp-num" style="--alvo: {alvo}" data-antes="{antes}" data-depois="{depois}"></div>
            <div class="lp-num-legenda">{legenda}</div>
        </div>""" for alvo, antes, depois, icone, cor, rotulo, legenda in NUMEROS)
    st.markdown(f'<div class="lp-numeros lp-entrada lp-d3">{blocos}</div>', unsafe_allow_html=True)


def _cabecalho_secao(numero: str, titulo: str, texto: str):
    st.markdown(f"""
    <div class="lp-secao lp-revelar">
        <div><div class="num">{numero}</div><h2>{titulo}</h2></div>
        <p>{texto}</p>
    </div>
    """, unsafe_allow_html=True)


def _cards():
    _cabecalho_secao("01 · Visão geral", "A ferramenta em quatro pontos",
                     "Do problema que motivou o trabalho ao que você pode testar agora.")
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
