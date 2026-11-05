import os
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from components.chart_builder import criar_grafico_movimento
from core.ambiente import MODELOS_PESADOS, padroes_execucao
from core.cache_resultados import carregar_resultado, chave_resultado, salvar_resultado
from core.catalogo_videos import (
    ROTULOS,
    ErroDownloadVideo,
    avaliar_resultado,
    carregar_catalogo,
    obter_video,
    recortar_quadro,
)
from core.eventos import AgrupadorEventos, chave_evento
from core.fonte_link import ErroLink, baixar_youtube, eh_youtube, validar_link
from core.heuristics import analisar_interacoes
from core.horario import hora_local
from core.serie_movimento import SerieMovimento
from core.tracker import PersonTracker
from core.video_processor import desenhar_anotacoes, redimensionar_frame
from core.video_saida import criar_gravador
from core.yolo_engine import MODELOS_TCC, carregar_modelo, processar_frame_detector, reiniciar_rastreamento
from pages_modules.camera_navegador import TRADUCOES, ParametrosAnalise, ProcessadorCamera, obter_configuracao_rtc
from pages_modules.constantes import (
    CHAVE_AMOSTRA_PADRAO,
    CHAVE_FONTE,
    CHAVE_MODO,
    FONTE_AMOSTRA,
    FONTE_LINK,
    FONTE_NAVEGADOR,
    FONTE_SERVIDOR,
    FONTE_UPLOAD,
    MODO_AB,
    MODO_COM,
    MODO_SEM,
)

# Muda quando a lógica da análise muda, para não reaproveitar resultados antigos do cache
VERSAO_ANALISE = 2
# Links e transmissões sem limite de quadros do modo leve param neste ponto (1 minuto a 30 FPS)
LIMITE_QUADROS_LINK = 1800
DURACAO_MAX_YOUTUBE_S = 180

TAXA_DE_ACERTO = ("No artigo (79 eventos), o YOLOv11s com velocidade teve precisão de 75,8% e revocação de 57,8%; "
                  "no UR Fall, 100% de precisão. O sistema pode errar, sobretudo em quedas na direção da câmera.")


def obter_modelo_da_sessao(model_name: str, acelerar: bool = False):
    """
    Carrega o modelo uma vez por sessão. O rastreador do Ultralytics guarda estado
    dentro do objeto do modelo, então compartilhá-lo entre sessões (st.cache_resource)
    misturaria IDs de usuários diferentes.
    """
    modelos = st.session_state.setdefault("modelos_carregados", {})
    chave = (model_name, acelerar)
    if chave not in modelos:
        modelos[chave] = carregar_modelo(model_name, acelerar=acelerar)
    return modelos[chave]


def salvar_upload_temporario(arquivo_upload) -> str:
    """
    Grava o upload em disco apenas quando o arquivo muda. O Streamlit reexecuta
    o script a cada interação, e sem isso cada clique criava outra cópia do vídeo.
    """
    identificador = getattr(arquivo_upload, "file_id", None) or (arquivo_upload.name, arquivo_upload.size)
    atual = st.session_state.get("upload_temporario")
    if atual and atual["id"] == identificador and os.path.isfile(atual["caminho"]):
        return atual["caminho"]

    if atual:
        try:
            os.remove(atual["caminho"])
        except OSError:
            pass

    sufixo = os.path.splitext(arquivo_upload.name)[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=sufixo) as tfile:
        tfile.write(arquivo_upload.getbuffer())
    st.session_state.upload_temporario = {"id": identificador, "caminho": tfile.name}
    return tfile.name


def registrar_eventos(eventos, agrupador, numero_frame, rotulo_modo, origem_rotulo, modelo_selecionado,
                      coletor=None, serie=None) -> int:
    """Adiciona ao histórico os eventos novos (agrupados por pessoa/dupla) e retorna quantas quedas entraram."""
    quedas = 0
    for ev in eventos:
        chave = chave_evento(ev)
        if chave is not None and not agrupador.novo_evento(chave, numero_frame):
            continue
        if ev["tipo"] == "QUEDA":
            quedas += 1
            if serie is not None:
                serie.registrar_queda(numero_frame, ev.get("id"))
        linha = {
            "Frame": numero_frame,
            "Timestamp": hora_local(),
            "Tipo": ev["tipo"],
            "Postura": ev["postura"],
            "Velocidade": ev["velocidade"],
            "Detalhes": ev["detalhes"],
            "Modo": rotulo_modo,
            "Fonte": origem_rotulo,
            "Modelo": modelo_selecionado
        }
        st.session_state.historico_eventos.append(linha)
        if coletor is not None:
            coletor.append(linha)
    return quedas


def painel_metricas(area, ritmo, pessoas, quedas_texto):
    """Telemetria compacta numa linha, para caber na coluna de controles sem rolar a tela."""
    area.markdown(
        f'<div class="estudio-metricas"><div><b>{ritmo:.1f}</b><span>quadros/s</span></div>'
        f'<div><b>{pessoas}</b><span>pessoas</span></div><div><b>{quedas_texto}</b><span>quedas</span></div></div>',
        unsafe_allow_html=True)


def status_ok(area, texto):
    area.markdown(f'<div class="estudio-status">{texto}</div>', unsafe_allow_html=True)


RESULTADO_GABARITO = {
    "VP": ("Verdadeiro positivo", "acerto: a queda foi detectada"),
    "VN": ("Verdadeiro negativo", "acerto: nenhuma queda registrada"),
    "FP": ("Falso positivo", "erro: alarme falso em atividade normal"),
    "FN": ("Falso negativo", "erro: a queda não foi detectada"),
}


def escolher_amostra():
    """Lista o catálogo do UR Fall e os vídeos locais de `videos/`; baixa a amostra escolhida se preciso.

    Retorna (caminho_video, origem_rotulo, item_do_catalogo ou None).
    """
    try:
        catalogo = carregar_catalogo()
    except (OSError, ValueError) as exc:
        st.warning(f"Não foi possível ler o catálogo de amostras ({exc}).")
        catalogo = {"videos": []}

    opcoes = {f"{v['nome']} — esperado: {ROTULOS[v['rotulo']].lower()}": v for v in catalogo["videos"]}
    pasta_videos = "videos"
    if os.path.isdir(pasta_videos):
        for arquivo in sorted(os.listdir(pasta_videos)):
            if arquivo.lower().endswith((".mp4", ".avi", ".mov")):
                opcoes[f"{arquivo} — vídeo local"] = os.path.join(pasta_videos, arquivo)

    if not opcoes:
        st.warning("Nenhuma amostra disponível. Faça o upload de um arquivo ou use a câmera do navegador.")
        return None, "Vídeo", None

    rotulos = list(opcoes)
    padrao = st.session_state.pop(CHAVE_AMOSTRA_PADRAO, None)  # definido pelo Get Started da página inicial
    if padrao:
        for rotulo, item in opcoes.items():
            if isinstance(item, dict) and item["id"] == padrao:
                st.session_state["amostra_escolhida"] = rotulo
    escolha = st.selectbox("Amostra", rotulos, key="amostra_escolhida")
    item = opcoes[escolha]
    if catalogo.get("citacao"):
        with st.expander("Sobre as amostras", expanded=False):
            st.caption(f"Vídeos do UR Fall Detection Dataset, usados para fins acadêmicos não comerciais conforme a "
                       f"licença do conjunto ({catalogo.get('licenca', 'CC BY-NC-SA 4.0')}). Fonte: "
                       f"{catalogo['citacao']} Disponível em: {catalogo.get('pagina', '')}. Cada vídeo traz a "
                       "profundidade e a imagem colorida lado a lado; a análise usa só a imagem colorida.")

    if isinstance(item, str):
        return item, os.path.basename(item), None

    try:
        with st.spinner(f"Baixando {item['arquivo']} do UR Fall Detection Dataset (na primeira vez)..."):
            caminho = obter_video(item)
    except ErroDownloadVideo as exc:
        st.warning(f"{exc} O site do UR Fall Detection Dataset pode estar fora do ar no momento. "
                   "Tente de novo mais tarde ou use a opção **Upload de Arquivo de Vídeo** com um vídeo próprio.")
        return None, item["nome"], None
    return str(caminho), item["nome"], item


def preparar_link(url: str, padroes):
    """Valida o link e, se for do YouTube, baixa o vídeo. Retorna (fonte para o OpenCV, rótulo) ou levanta ErroLink."""
    url = validar_link(url)
    if not eh_youtube(url):
        return url, url
    limite = DURACAO_MAX_YOUTUBE_S
    anterior = st.session_state.get("youtube_baixado")
    if anterior and anterior["url"] == url and os.path.isfile(anterior["caminho"]):
        return anterior["caminho"], url
    if anterior:
        shutil.rmtree(anterior["pasta"], ignore_errors=True)
    pasta = tempfile.mkdtemp(prefix="dq_youtube_")
    with st.spinner("Baixando o vídeo do YouTube..."):
        caminho = baixar_youtube(url, Path(pasta), max_duracao_s=limite)
    st.session_state.youtube_baixado = {"url": url, "caminho": str(caminho), "pasta": pasta}
    return str(caminho), url


def mostrar_gabarito(item: dict, resultados):
    """Compara o resultado de cada modo com o rótulo esperado da amostra do catálogo."""
    esperado = ROTULOS[item["rotulo"]]
    linhas = []
    for modo, quedas in resultados:
        sigla = avaliar_resultado(item["rotulo"], quedas > 0)
        nome, explicacao = RESULTADO_GABARITO[sigla]
        sistema = f"{quedas} queda(s)" if quedas else "nenhuma queda"
        classe = "acerto" if sigla in ("VP", "VN") else "erro"
        linhas.append(f'<div class="gabarito-linha {classe}"><span class="modo">{modo}</span>'
                      f'<span>{sistema}</span><span class="sigla" title="{nome}: {explicacao}">{sigla}</span></div>')
    st.markdown(f'<div class="gabarito"><div class="gabarito-titulo">Gabarito · esperado: <b>{esperado}</b></div>'
                f'{"".join(linhas)}</div>', unsafe_allow_html=True)


def mostrar_grafico(area, serie, limiar_ar, limiar_vel, chave="grafico_final"):
    # Cada redesenho no mesmo espaço precisa de uma chave própria, senão o Streamlit acusa elemento duplicado
    if serie and serie.get("frame"):
        area.plotly_chart(criar_grafico_movimento(serie, limiar_ar, limiar_vel), width="stretch",
                          config={"displayModeBar": False}, key=chave)


def mostrar_resultado(dados: dict, video, amostra, modo_execucao: str, area_video, area_grafico, area_resultado,
                      limiar_ar: float, limiar_vel: float):
    """Exibe o vídeo anotado em velocidade real, o gráfico de movimento e, para amostras, o gabarito."""
    if video:
        with area_video.container():
            st.video(Path(video).read_bytes(), format="video/mp4")
            st.caption("Vídeo anotado em velocidade real, com as caixas, a postura e os alertas da análise.")
    mostrar_grafico(area_grafico, dados.get("serie"), limiar_ar, limiar_vel)
    if dados.get("concluido") and amostra:
        resultados = []
        if modo_execucao in (MODO_SEM, MODO_AB):
            resultados.append(("Sem velocidade", dados["quedas_sem"]))
        if modo_execucao in (MODO_COM, MODO_AB):
            resultados.append(("Com velocidade", dados["quedas_com"]))
        with area_resultado.container():
            mostrar_gabarito(amostra, resultados)


def render_camera_navegador(parametros: ParametrosAnalise, modo_execucao: str, col_controles, col_video):
    """Análise ao vivo da câmera de quem acessa a página, via WebRTC."""
    configuracao_rtc, com_turn, erro_turn = obter_configuracao_rtc()
    with col_controles:
        if com_turn:
            st.caption("O navegador pedirá permissão de câmera. A conexão usa um servidor TURN para atravessar "
                       "firewalls e redes com NAT.")
        else:
            st.caption("O navegador pedirá permissão de câmera. Sem servidor TURN configurado, a conexão só funciona "
                       "em redes abertas (como na execução local); veja o README.")
        if erro_turn:
            st.caption(f"Não foi possível obter as credenciais TURN ({erro_turn}); usando apenas STUN.")
        area_metricas = st.empty()
        alerta = st.empty()
        tabela = st.empty()

    with st.spinner("Carregando o modelo..."):
        modelo = obter_modelo_da_sessao(parametros.modelo, padroes_execucao().usar_openvino)

    with col_video:
        ctx = webrtc_streamer(
            key=f"camera_navegador_{parametros.modelo}",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=configuracao_rtc,
            # Resolução e taxa moderadas: menos quadros para decodificar e reenviar ao navegador
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                                "frameRate": {"ideal": 15, "max": 20}}, "audio": False},
            video_processor_factory=lambda: ProcessadorCamera(modelo, parametros),
            async_processing=True,
            translations=TRADUCOES,
        )
        if modo_execucao == MODO_AB:
            st.caption("Esquerda: sem velocidade (linha de base). Direita: com velocidade (proposta).")

    while ctx.state.playing:
        processador = ctx.video_processor
        if processador is not None:
            processador.atualizar_parametros(parametros)
            while not processador.eventos.empty():
                st.session_state.historico_eventos.append(processador.eventos.get_nowait())
            m = processador.metricas()
            if modo_execucao == MODO_AB:
                texto = f"{m['quedas_com']} / {m['quedas_sem']}"
            else:
                texto = str(m["quedas_com"] if parametros.executar_com else m["quedas_sem"])
            painel_metricas(area_metricas, m["fps"], m["pessoas"], texto)
            if m["ultima_queda_frame"] is not None and m["frame"] - m["ultima_queda_frame"] < 60:
                alerta.error(f"**ALERTA DE QUEDA NO QUADRO {m['ultima_queda_frame']}!**")
            else:
                alerta.empty()
            recentes = [e for e in st.session_state.historico_eventos if e["Fonte"] == "Câmera do navegador"]
            if recentes:
                tabela.dataframe(recentes[-5:][::-1], width="stretch", hide_index=True,
                                 column_order=("Timestamp", "Tipo", "Detalhes", "Modo"))
        time.sleep(0.5)


def _parametros_barra_lateral(padroes):
    st.sidebar.header("Parâmetros de Inferência")
    modelo_selecionado = st.sidebar.selectbox(
        "Modelo YOLOv11",
        list(MODELOS_TCC.keys()),
        index=1,  # Default YOLOv11s (o sweet spot apontado no artigo)
        help="Selecione a variante da arquitetura YOLOv11."
    )
    if padroes.modo_leve and any(m in modelo_selecionado for m in MODELOS_PESADOS):
        st.sidebar.markdown(
            '<div class="aviso-discreto" lang="pt-BR">Modelos grandes (L e X) usam muita CPU e podem fazer a '
            "plataforma reduzir temporariamente o desempenho do app. Na versão publicada, prefira v11n ou v11s.</div>",
            unsafe_allow_html=True,
        )

    if st.session_state.get(CHAVE_MODO) not in (MODO_COM, MODO_SEM, MODO_AB):
        st.session_state[CHAVE_MODO] = MODO_COM
    modo_execucao = st.sidebar.radio(
        "Modo de Análise",
        [MODO_COM, MODO_SEM, MODO_AB],
        key=CHAVE_MODO,
        help="Compare a filtragem de falsos positivos da heurística cinemática contra o modelo estático."
    )

    with st.sidebar.expander("Calibração de Limiares Heurísticos", expanded=False):
        conf_min = st.slider("Confiança Mínima YOLO", 0.1, 0.9, 0.5, 0.05)
        limiar_ar = st.slider("Limiar de Razão de Aspecto (AR Queda)", 0.5, 1.0, 0.75, 0.05,
                              help="Valores abaixo deste limiar caracterizam o corpo na horizontal (Deitado).")
        limiar_vel = st.slider("Limiar de Velocidade de Queda (px/f)", 5.0, 40.0, 15.0, 1.0,
                               help="Velocidade mínima do centroide para confirmar a transição brusca de queda.")
        limiar_prox = st.slider("Distância de Proximidade (px)", 20, 150, 60, 5,
                                help="Distância euclidiana máxima entre centroides para disparar alerta de aglomeração.")
        resolucao_alvo = st.select_slider(
            "Resolução de Processamento",
            options=[480, 720, 1080],
            value=1080,
            help="1080p é o padrão da metodologia do TCC; 720p ou 480p aceleram a execução em CPU."
        )
        corrigir_fps = st.checkbox(
            "Normalizar velocidade pelo FPS do vídeo (referência 30 FPS)",
            value=True,
            help="Mantém o limiar em px/f com o mesmo significado em vídeos de 25, 30 ou 60 FPS."
        )
        carencia_s = st.slider(
            "Agrupar ocorrências em eventos (s)", 0.0, 5.0, 2.0, 0.5,
            help="A mesma pessoa (ou dupla) só gera um novo registro após este tempo sem ocorrência. 0 registra todo frame."
        )
        pular_quadros = st.slider(
            "Analisar 1 a cada N quadros", 1, 5, padroes.pular_quadros,
            help="1 analisa todos os quadros (metodologia do TCC). Valores maiores reduzem o uso de CPU, mas "
                 "a velocidade passa a ser a média entre os quadros analisados e o pico da queda fica diluído: "
                 "quedas rápidas podem deixar de ser detectadas."
        )
        if pular_quadros > 1:
            st.warning("Pulando quadros, o resultado pode diferir da metodologia do TCC.")
    return (modelo_selecionado, modo_execucao, conf_min, limiar_ar, limiar_vel, limiar_prox, resolucao_alvo,
            corrigir_fps, carencia_s, pular_quadros)


def _dica_modo_leve(padroes):
    texto = ("<b>Versão publicada (Streamlit Cloud gratuito).</b> Para não sobrecarregar o servidor, a análise usa "
             f"um modo leve, com até {padroes.max_quadros} quadros por vídeo e imagens menores na tela. Se a "
             "análise ficar lenta ou aparecer o aviso em inglês <i>Your app has been throttled</i>, é a plataforma "
             "reduzindo a CPU do app por alguns minutos por causa do uso intenso; o app continua funcionando e o "
             "aviso some sozinho. Modelos menores (v11n, v11s) evitam o problema.")
    st.markdown(
        '<div class="info-tooltip-linha"><span class="info-tooltip" tabindex="0" role="button" '
        'aria-label="Sobre o modo leve da versão publicada" aria-describedby="dica-modo-leve">'
        '<span class="info-tooltip-icone" aria-hidden="true">info</span>Modo leve'
        f'<span class="info-tooltip-texto" role="tooltip" id="dica-modo-leve" lang="pt-BR">{texto}</span></span></div>',
        unsafe_allow_html=True,
    )


def render_estudio_inferencia():
    st.markdown('<div class="estudio-cabecalho"><span class="titulo">Estúdio de Inferência</span>'
                '<span class="sub">Detecção de quedas com YOLOv11 e heurística de postura e velocidade</span></div>',
                unsafe_allow_html=True)

    padroes = padroes_execucao()
    (modelo_selecionado, modo_execucao, conf_min, limiar_ar, limiar_vel, limiar_prox, resolucao_alvo,
     corrigir_fps, carencia_s, pular_quadros) = _parametros_barra_lateral(padroes)

    if "historico_eventos" not in st.session_state:
        st.session_state.historico_eventos = []

    # Tela única: controles e resultado à esquerda, vídeo e gráfico à direita
    col_controles, col_video = st.columns([1, 1.55], gap="large")

    with col_controles:
        fontes = [FONTE_AMOSTRA, FONTE_UPLOAD, FONTE_LINK, FONTE_NAVEGADOR, FONTE_SERVIDOR]
        if st.session_state.get(CHAVE_FONTE) not in fontes:
            st.session_state[CHAVE_FONTE] = FONTE_AMOSTRA
        tipo_fonte = st.selectbox("Fonte de entrada", fontes, key=CHAVE_FONTE)

    if tipo_fonte == FONTE_NAVEGADOR:
        render_camera_navegador(ParametrosAnalise(
            executar_com=modo_execucao in (MODO_COM, MODO_AB),
            executar_sem=modo_execucao in (MODO_SEM, MODO_AB),
            conf_min=conf_min, limiar_ar=limiar_ar, limiar_vel=limiar_vel, limiar_prox=limiar_prox,
            resolucao_alvo=resolucao_alvo, corrigir_fps=corrigir_fps, carencia_s=carencia_s,
            modelo=modelo_selecionado,
        ), modo_execucao, col_controles, col_video)
        return

    caminho_video = None
    origem_rotulo = "Vídeo"
    amostra_catalogo = None
    url_link = ""

    with col_controles:
        if tipo_fonte == FONTE_UPLOAD:
            arquivo_upload = st.file_uploader("Vídeo (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
            if arquivo_upload is not None:
                caminho_video = salvar_upload_temporario(arquivo_upload)
                origem_rotulo = arquivo_upload.name
        elif tipo_fonte == FONTE_AMOSTRA:
            caminho_video, origem_rotulo, amostra_catalogo = escolher_amostra()
        elif tipo_fonte == FONTE_LINK:
            url_link = st.text_input(
                "Endereço do vídeo ou da transmissão", placeholder="https://… (YouTube, MP4, HLS) ou rtsp://…",
                help="Aceita links do YouTube (até 3 minutos), arquivos de vídeo na web e transmissões HLS, RTSP ou "
                     "RTMP, como as do OBS, do VLC ou de câmeras IP. Transmissões ao vivo são analisadas até você "
                     "clicar em Parar ou até o limite de quadros.")
            origem_rotulo = "Link"
        elif tipo_fonte == FONTE_SERVIDOR:
            st.caption("Usa a câmera do computador que executa o Streamlit. Serve apenas quando o sistema roda na "
                       "sua própria máquina; na versão publicada, use a Câmera do Navegador.")
            caminho_video = 0
            origem_rotulo = "Webcam do servidor"

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            btn_iniciar = st.button("Iniciar Análise", type="primary", icon=":material/play_arrow:", width="stretch")
        with col_btn2:
            # Qualquer clique reexecuta o script e interrompe o laço em andamento
            st.button("Parar", icon=":material/stop:", width="stretch")

        area_status = st.empty()
        barra = st.empty()
        area_metricas = st.empty()
        alert_placeholder = st.empty()
        area_resultado = st.empty()

        st.markdown(f'<div class="taxa-acerto">{TAXA_DE_ACERTO}</div>', unsafe_allow_html=True)
        col_dica, col_limpar = st.columns([1, 1], vertical_alignment="center")
        with col_limpar:
            if st.button("Limpar ocorrências", icon=":material/delete:", type="tertiary", width="stretch"):
                st.session_state.historico_eventos = []
                st.rerun()
        with col_dica:
            if padroes.modo_leve:
                _dica_modo_leve(padroes)

    with col_video:
        area_video = st.empty()
        area_grafico = st.empty()

    if not btn_iniciar:
        with area_video.container():
            st.markdown('<div class="estudio-vazio"><span class="icone">play_circle</span>'
                        '<div>Escolha a fonte e clique em <b>Iniciar Análise</b>.<br>O vídeo anotado e o gráfico de '
                        'postura e velocidade aparecem aqui.</div></div>', unsafe_allow_html=True)
        return

    if tipo_fonte == FONTE_LINK:
        try:
            caminho_video, origem_rotulo = preparar_link(url_link, padroes)
        except ErroLink as exc:
            area_status.warning(str(exc))
            return

    if caminho_video is None:
        area_status.error("Selecione uma fonte de vídeo válida antes de iniciar.")
        return

    executar_com = modo_execucao in (MODO_COM, MODO_AB)
    executar_sem = modo_execucao in (MODO_SEM, MODO_AB)

    def exibir_metricas(ritmo, pessoas, quedas_com, quedas_sem):
        if modo_execucao == MODO_AB:
            texto = f"{quedas_com} / {quedas_sem}"
        else:
            texto = str(quedas_com if executar_com else quedas_sem)
        painel_metricas(area_metricas, ritmo, pessoas, texto)

    # Amostras do catálogo: uma análise idêntica já feita é reaproveitada na hora
    chave_cache = None
    if amostra_catalogo:
        chave_cache = chave_resultado(
            versao=VERSAO_ANALISE, video=amostra_catalogo["id"], modelo=modelo_selecionado, modo=modo_execucao,
            conf=conf_min, ar=limiar_ar, vel=limiar_vel, prox=limiar_prox, largura=resolucao_alvo,
            corrigir_fps=corrigir_fps, carencia=carencia_s, pular=pular_quadros, max_quadros=padroes.max_quadros,
            openvino=padroes.usar_openvino,
        )
        guardado = carregar_resultado(chave_cache)
        if guardado:
            dados, video = guardado
            for linha in dados["eventos"]:
                st.session_state.historico_eventos.append({**linha, "Timestamp": hora_local()})
            status_ok(area_status, f"Resultado reaproveitado de uma análise idêntica: {dados['numero_frame']} "
                                   f"quadros (a primeira análise levou {dados['tempo_s']:.0f} s).")
            exibir_metricas(dados["numero_frame"] / max(0.001, dados["tempo_s"]), dados.get("pessoas", 0),
                            dados["quedas_com"], dados["quedas_sem"])
            mostrar_resultado(dados, video, amostra_catalogo, modo_execucao, area_video, area_grafico,
                              area_resultado, limiar_ar, limiar_vel)
            return

    with st.spinner("Inicializando modelo e preparando pipeline..."):
        modelo = obter_modelo_da_sessao(modelo_selecionado, padroes.usar_openvino)
    reiniciar_rastreamento(modelo)

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        if caminho_video == 0:
            area_status.error("Não foi possível abrir uma câmera neste servidor. Isso é esperado na versão publicada "
                              "(Streamlit Cloud), que roda em um servidor sem câmera. Escolha **Câmera do Navegador** "
                              "para usar a câmera do seu dispositivo.")
        elif tipo_fonte == FONTE_LINK:
            area_status.error("Não foi possível abrir o link. Confira se o endereço está correto, se a transmissão "
                              "está no ar e se ela é acessível a partir deste servidor.")
        else:
            area_status.error("Não foi possível abrir o vídeo selecionado. Verifique se o arquivo não está corrompido "
                              "e se o formato é .mp4, .avi ou .mov.")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    if not fps_video or fps_video <= 0 or fps_video > 240:
        fps_video = 30.0  # transmissões nem sempre informam a taxa de quadros
    tracker = PersonTracker(fps_video=fps_video if corrigir_fps else None, passo_quadros=pular_quadros)
    carencia_frames = round(fps_video * carencia_s)
    agrupador_com = AgrupadorEventos(carencia_frames)
    agrupador_sem = AgrupadorEventos(carencia_frames)

    max_quadros = padroes.max_quadros
    if tipo_fonte in (FONTE_LINK, FONTE_SERVIDOR) and not max_quadros:
        max_quadros = LIMITE_QUADROS_LINK
    total_quadros = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_quadros <= 0:
        total_quadros = 0  # transmissão ao vivo: sem total conhecido
    if max_quadros:
        total_quadros = min(total_quadros, max_quadros) if total_quadros else max_quadros

    barra.progress(0.0, text="Preparando a análise...")
    with area_video.container():
        if modo_execucao == MODO_AB:
            col_v1, col_v2 = st.columns(2)
            view_left = col_v1.empty()
            view_right = col_v2.empty()
        else:
            view_main = st.empty()
        st.caption("Prévia da análise, atualizada algumas vezes por segundo. Ao final, o vídeo anotado é exibido "
                   "na velocidade real do vídeo original.")

    numero_frame = 0
    quedas_com = 0
    quedas_sem = 0
    inicio_tempo = time.time()
    recorte = amostra_catalogo.get("recorte") if amostra_catalogo else None
    concluido = False
    limite_atingido = False
    quadros_analisados = 0
    eventos_gerados = []
    serie = SerieMovimento()
    pessoas = []
    ultima_atualizacao = 0.0
    ultimo_grafico = 0.0
    largura_video = padroes.largura_exibicao * (2 if modo_execucao == MODO_AB else 1)
    anterior = st.session_state.get("pasta_video_resultado")
    if anterior:
        shutil.rmtree(anterior, ignore_errors=True)  # o vídeo da análise anterior desta sessão já foi exibido
    pasta_video = tempfile.mkdtemp(prefix="dq_resultado_")
    st.session_state.pasta_video_resultado = pasta_video
    arquivo_video = Path(pasta_video) / "resultado.mp4"
    gravador = criar_gravador(arquivo_video, fps_video / pular_quadros, largura_max=largura_video)

    def exibir(imagem):
        """Reduz a imagem enviada ao navegador; a análise continua na resolução escolhida."""
        altura, largura = imagem.shape[:2]
        if largura > padroes.largura_exibicao:
            imagem = cv2.resize(imagem, (padroes.largura_exibicao, round(altura * padroes.largura_exibicao / largura)))
        return cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)

    try:
        while cap.isOpened():
            if max_quadros and numero_frame >= max_quadros:
                limite_atingido = True
                break
            # Quadros pulados só avançam o vídeo (grab), sem decodificar nem rodar o YOLO
            if pular_quadros > 1 and numero_frame % pular_quadros:
                if not cap.grab():
                    concluido = True
                    break
                numero_frame += 1
                continue

            ret, frame = cap.read()
            if not ret:
                concluido = True
                break

            numero_frame += 1
            quadros_analisados += 1
            frame = redimensionar_frame(recortar_quadro(frame, recorte), largura_alvo=resolucao_alvo)

            # Uma única inferência por frame alimenta os dois ramos heurísticos
            pessoas = processar_frame_detector(modelo, frame, tracker=tracker, confianca=conf_min)
            alertas_prox = analisar_interacoes([p["centroide"] for p in pessoas], distancia_minima=limiar_prox)
            serie.registrar(numero_frame, pessoas)

            queda_no_frame = False
            if executar_com:
                frame_com, eventos_com = desenhar_anotacoes(
                    frame, pessoas, alertas_prox,
                    modo_velocidade=True, limiar_queda=limiar_vel, limiar_ar=limiar_ar
                )
                quedas_com += registrar_eventos(eventos_com, agrupador_com, numero_frame, "Com Velocidade",
                                                origem_rotulo, modelo_selecionado, eventos_gerados, serie)
                queda_no_frame = queda_no_frame or any(e["tipo"] == "QUEDA" for e in eventos_com)
            if executar_sem:
                frame_sem, eventos_sem = desenhar_anotacoes(
                    frame, pessoas, alertas_prox,
                    modo_velocidade=False, limiar_queda=limiar_vel, limiar_ar=limiar_ar
                )
                # No A/B, o gráfico marca as quedas da proposta (com velocidade)
                quedas_sem += registrar_eventos(eventos_sem, agrupador_sem, numero_frame, "Sem Velocidade",
                                                origem_rotulo, modelo_selecionado, eventos_gerados,
                                                None if executar_com else serie)
                if modo_execucao == MODO_SEM:
                    queda_no_frame = any(e["tipo"] == "QUEDA" for e in eventos_sem)

            if gravador is not None:
                if modo_execucao == MODO_AB:
                    gravador.escrever(cv2.hconcat([frame_sem, frame_com]))
                else:
                    gravador.escrever(frame_sem if modo_execucao == MODO_SEM else frame_com)

            if queda_no_frame:
                alert_placeholder.error(f"**ALERTA DE QUEDA NO QUADRO {numero_frame}!**")

            # A prévia e a telemetria são atualizadas ~4 vezes por segundo e o gráfico ~1 vez por segundo:
            # enviar cada quadro ao navegador custa CPU e não muda o resultado
            agora = time.time()
            if agora - ultima_atualizacao >= 0.25:
                ultima_atualizacao = agora
                if modo_execucao == MODO_AB:
                    view_left.image(exibir(frame_sem), caption="Sem velocidade", width="stretch")
                    view_right.image(exibir(frame_com), caption="Com velocidade", width="stretch")
                elif modo_execucao == MODO_SEM:
                    view_main.image(exibir(frame_sem), caption=f"Modo Estático: Frame {numero_frame}", width="stretch")
                else:
                    view_main.image(exibir(frame_com), caption=f"Modo Bimodal: Frame {numero_frame}", width="stretch")

                ritmo = quadros_analisados / max(0.001, agora - inicio_tempo)
                exibir_metricas(ritmo, len(pessoas), quedas_com, quedas_sem)
                if total_quadros:
                    restante = max(0.0, (total_quadros - numero_frame) / pular_quadros / max(ritmo, 0.01))
                    barra.progress(min(1.0, numero_frame / total_quadros),
                                   text=f"Quadro {numero_frame} de {total_quadros} · cerca de {restante:.0f} s restantes")
                else:
                    barra.progress(0.0, text=f"Quadro {numero_frame} · transmissão ao vivo (clique em Parar)")
            if agora - ultimo_grafico >= 1.0:
                ultimo_grafico = agora
                mostrar_grafico(area_grafico, serie.dados_principal(), limiar_ar, limiar_vel,
                                chave=f"grafico_{numero_frame}")

            if padroes.pausa_s:
                time.sleep(padroes.pausa_s)  # pausa cooperativa: deixa a CPU respirar entre quadros

    except Exception as exc:  # dependência ausente no servidor, vídeo corrompido etc.
        barra.empty()
        area_status.error(
            f"A análise foi interrompida no frame {numero_frame} por um erro no processamento: "
            f"**{type(exc).__name__}: {exc}**. Se o erro citar um módulo ausente, a dependência precisa estar "
            "no `requirements.txt` do app publicado."
        )
        return
    finally:
        cap.release()
        video = gravador.fechar() if gravador is not None else None

    tempo_total = time.time() - inicio_tempo
    barra.empty()
    detalhe = "" if pular_quadros == 1 else f", {quadros_analisados} analisados (1 a cada {pular_quadros})"
    status_ok(area_status, f"Concluído: {numero_frame} quadros{detalhe} em {tempo_total:.1f} s.")
    exibir_metricas(quadros_analisados / max(0.001, tempo_total), len(pessoas), quedas_com, quedas_sem)
    if limite_atingido:
        alert_placeholder.warning(f"A análise parou no limite de {max_quadros} quadros. Para vídeos longos, rode o "
                                  "sistema localmente (veja o README).")

    dados = {"numero_frame": numero_frame, "tempo_s": tempo_total, "quedas_com": quedas_com,
             "quedas_sem": quedas_sem, "concluido": concluido, "eventos": eventos_gerados,
             "pessoas": len(pessoas), "serie": serie.dados_principal()}
    if concluido and chave_cache:
        try:
            salvar_resultado(chave_cache, dados, video)
        except OSError:
            pass  # sem espaço em disco, por exemplo: só não haverá reaproveitamento
    # Sem vídeo (falha na gravação), a prévia do último quadro continua na tela como reserva
    mostrar_resultado(dados, video, amostra_catalogo, modo_execucao, area_video, area_grafico, area_resultado,
                      limiar_ar, limiar_vel)
