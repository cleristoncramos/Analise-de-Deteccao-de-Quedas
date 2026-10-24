import os
import tempfile
import time

import cv2
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from core.eventos import AgrupadorEventos, chave_evento
from core.heuristics import analisar_interacoes
from core.tracker import PersonTracker
from core.video_processor import desenhar_anotacoes, redimensionar_frame
from core.yolo_engine import MODELOS_TCC, carregar_modelo, processar_frame_detector, reiniciar_rastreamento
from pages_modules.camera_navegador import TRADUCOES, ParametrosAnalise, ProcessadorCamera, obter_configuracao_rtc

MODO_COM = "Bimodal (Com Velocidade - Proposta)"
MODO_SEM = "Estático (Sem Velocidade - Linha de Base)"
MODO_AB = "Comparativo Lado a Lado (A/B: Sem vs Com Velocidade)"

FONTE_UPLOAD = "Upload de Arquivo de Vídeo"
FONTE_AMOSTRA = "Amostra Pré-Carregada"
FONTE_NAVEGADOR = "Câmera do Navegador"
FONTE_SERVIDOR = "Webcam do Servidor (uso local)"


def obter_modelo_da_sessao(model_name: str):
    """
    Carrega o modelo uma vez por sessão. O rastreador do Ultralytics guarda estado
    dentro do objeto do modelo, então compartilhá-lo entre sessões (st.cache_resource)
    misturaria IDs de usuários diferentes.
    """
    modelos = st.session_state.setdefault("modelos_carregados", {})
    if model_name not in modelos:
        modelos[model_name] = carregar_modelo(model_name)
    return modelos[model_name]


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


def registrar_eventos(eventos, agrupador, numero_frame, rotulo_modo, origem_rotulo, modelo_selecionado) -> int:
    """Adiciona ao histórico os eventos novos (agrupados por pessoa/dupla) e retorna quantas quedas entraram."""
    quedas = 0
    for ev in eventos:
        chave = chave_evento(ev)
        if chave is not None and not agrupador.novo_evento(chave, numero_frame):
            continue
        if ev["tipo"] == "QUEDA":
            quedas += 1
        st.session_state.historico_eventos.append({
            "Frame": numero_frame,
            "Timestamp": time.strftime("%H:%M:%S"),
            "Tipo": ev["tipo"],
            "Postura": ev["postura"],
            "Velocidade": ev["velocidade"],
            "Detalhes": ev["detalhes"],
            "Modo": rotulo_modo,
            "Fonte": origem_rotulo,
            "Modelo": modelo_selecionado
        })
    return quedas


def render_camera_navegador(parametros: ParametrosAnalise, modo_execucao: str):
    """Análise ao vivo da câmera de quem acessa a página, via WebRTC."""
    configuracao_rtc, com_turn, erro_turn = obter_configuracao_rtc()
    if com_turn:
        st.caption("Ao ligar a câmera, o navegador pedirá permissão de acesso. A conexão usa um servidor TURN "
                   "para atravessar firewalls e redes com NAT.")
    else:
        st.caption("Ao ligar a câmera, o navegador pedirá permissão de acesso. Sem servidor TURN configurado, "
                   "a conexão só funciona em redes abertas (como na execução local); no app publicado é preciso "
                   "configurar o TURN nos Secrets (veja o README).")
    if erro_turn:
        st.caption(f"Não foi possível obter as credenciais TURN ({erro_turn}); usando apenas STUN.")

    with st.spinner("Carregando o modelo..."):
        modelo = obter_modelo_da_sessao(parametros.modelo)

    ctx = webrtc_streamer(
        key=f"camera_navegador_{parametros.modelo}",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=configuracao_rtc,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: ProcessadorCamera(modelo, parametros),
        async_processing=True,
        translations=TRADUCOES,
    )
    if modo_execucao == MODO_AB:
        st.caption("Imagem da esquerda: sem velocidade (linha de base). Direita: com velocidade (proposta).")

    col1, col2, col3 = st.columns(3)
    m_fps, m_pessoas, m_quedas = col1.empty(), col2.empty(), col3.empty()
    alerta = st.empty()
    tabela = st.empty()

    while ctx.state.playing:
        processador = ctx.video_processor
        if processador is not None:
            processador.atualizar_parametros(parametros)
            while not processador.eventos.empty():
                st.session_state.historico_eventos.append(processador.eventos.get_nowait())
            m = processador.metricas()
            m_fps.metric("FPS", f"{m['fps']:.1f}")
            m_pessoas.metric("Pessoas em Cena", m["pessoas"])
            if modo_execucao == MODO_AB:
                m_quedas.metric("Quedas (Com / Sem Vel.)", f"{m['quedas_com']} / {m['quedas_sem']}")
            else:
                m_quedas.metric("Quedas Registradas",
                                m["quedas_com"] if parametros.executar_com else m["quedas_sem"])
            if m["ultima_queda_frame"] is not None and m["frame"] - m["ultima_queda_frame"] < 60:
                alerta.error(f"**ALERTA CRÍTICO DE QUEDA DETECTADO NO FRAME {m['ultima_queda_frame']}!**")
            else:
                alerta.empty()
            recentes = [e for e in st.session_state.historico_eventos if e["Fonte"] == "Câmera do navegador"]
            if recentes:
                tabela.dataframe(recentes[-8:][::-1], width="stretch", hide_index=True)
        time.sleep(0.5)


def render_estudio_inferencia():
    st.title("Estúdio de Inferência e Detecção")
    st.caption("Bancada de testes interativa para execução dos modelos YOLOv11 com análise cinemática e postural")

    # Barra lateral de configurações
    st.sidebar.header("Parâmetros de Inferência")

    modelo_selecionado = st.sidebar.selectbox(
        "Modelo YOLOv11",
        list(MODELOS_TCC.keys()),
        index=1,  # Default YOLOv11s (o sweet spot apontado no artigo)
        help="Selecione a variante da arquitetura YOLOv11."
    )

    modo_execucao = st.sidebar.radio(
        "Modo de Análise",
        [MODO_COM, MODO_SEM, MODO_AB],
        index=0,
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

    # Seleção da fonte de vídeo
    tipo_fonte = st.radio(
        "Fonte de Entrada",
        [FONTE_UPLOAD, FONTE_AMOSTRA, FONTE_NAVEGADOR, FONTE_SERVIDOR],
        horizontal=True
    )

    # Inicialização do histórico de eventos na sessão
    if "historico_eventos" not in st.session_state:
        st.session_state.historico_eventos = []

    if tipo_fonte == FONTE_NAVEGADOR:
        render_camera_navegador(ParametrosAnalise(
            executar_com=modo_execucao in (MODO_COM, MODO_AB),
            executar_sem=modo_execucao in (MODO_SEM, MODO_AB),
            conf_min=conf_min, limiar_ar=limiar_ar, limiar_vel=limiar_vel, limiar_prox=limiar_prox,
            resolucao_alvo=resolucao_alvo, corrigir_fps=corrigir_fps, carencia_s=carencia_s,
            modelo=modelo_selecionado,
        ), modo_execucao)
        return

    caminho_video = None
    origem_rotulo = "Vídeo"

    if tipo_fonte == FONTE_UPLOAD:
        arquivo_upload = st.file_uploader("Selecione um vídeo (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if arquivo_upload is not None:
            caminho_video = salvar_upload_temporario(arquivo_upload)
            origem_rotulo = arquivo_upload.name

    elif tipo_fonte == FONTE_AMOSTRA:
        pasta_videos = "videos"
        os.makedirs(pasta_videos, exist_ok=True)
        amostras = sorted(f for f in os.listdir(pasta_videos) if f.lower().endswith((".mp4", ".avi", ".mov")))
        if amostras:
            video_escolhido = st.selectbox("Escolha uma amostra disponível:", amostras)
            caminho_video = os.path.join(pasta_videos, video_escolhido)
            origem_rotulo = video_escolhido
        else:
            st.warning("Nenhum vídeo encontrado na pasta `videos/`. Faça o upload de um arquivo ou use a câmera do navegador.")

    elif tipo_fonte == FONTE_SERVIDOR:
        st.caption("Usa a câmera do computador que executa o Streamlit. Serve apenas quando o sistema roda na "
                   "sua própria máquina; na versão publicada, use a Câmera do Navegador.")
        caminho_video = 0
        origem_rotulo = "Webcam do servidor"

    st.divider()

    # Controles de Execução
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        btn_iniciar = st.button("Iniciar Análise", type="primary", icon=":material/play_arrow:", width="stretch")
    with col_btn2:
        # Qualquer clique reexecuta o script e interrompe o laço em andamento
        st.button("Parar", icon=":material/stop:", width="stretch")
    with col_btn3:
        btn_limpar_logs = st.button("Limpar Ocorrências", icon=":material/delete:", width="stretch")
        if btn_limpar_logs:
            st.session_state.historico_eventos = []
            st.rerun()

    if not btn_iniciar:
        return

    if caminho_video is None:
        st.error("Por favor, selecione uma fonte de vídeo válida antes de iniciar.")
        return

    with st.spinner("Inicializando modelo e preparando pipeline..."):
        modelo = obter_modelo_da_sessao(modelo_selecionado)
    reiniciar_rastreamento(modelo)

    cap = cv2.VideoCapture(caminho_video)
    if not cap.isOpened():
        if caminho_video == 0:
            st.error("Não foi possível abrir uma câmera neste servidor. Isso é esperado na versão publicada "
                     "(Streamlit Cloud), que roda em um servidor sem câmera. Escolha **Câmera do Navegador** "
                     "para usar a câmera do seu dispositivo.")
        else:
            st.error("Não foi possível abrir o vídeo selecionado. Verifique se o arquivo não está corrompido "
                     "e se o formato é .mp4, .avi ou .mov.")
        return

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    tracker = PersonTracker(fps_video=fps_video if corrigir_fps else None)

    executar_com = modo_execucao in (MODO_COM, MODO_AB)
    executar_sem = modo_execucao in (MODO_SEM, MODO_AB)
    carencia_frames = round((fps_video if fps_video and fps_video > 0 else 30.0) * carencia_s)
    agrupador_com = AgrupadorEventos(carencia_frames)
    agrupador_sem = AgrupadorEventos(carencia_frames)

    # Placeholders de visualização
    status_placeholder = st.empty()
    telemetria_col1, telemetria_col2, telemetria_col3 = st.columns(3)
    m_fps = telemetria_col1.empty()
    m_pessoas = telemetria_col2.empty()
    m_quedas = telemetria_col3.empty()

    if modo_execucao == MODO_AB:
        col_v1, col_v2 = st.columns(2)
        view_left = col_v1.empty()
        view_right = col_v2.empty()
    else:
        view_main = st.empty()

    alert_placeholder = st.empty()

    numero_frame = 0
    quedas_com = 0
    quedas_sem = 0
    inicio_tempo = time.time()

    try:
        while cap.isOpened():
            t_frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            numero_frame += 1
            frame = redimensionar_frame(frame, largura_alvo=resolucao_alvo)

            # Uma única inferência por frame alimenta os dois ramos heurísticos
            pessoas = processar_frame_detector(modelo, frame, tracker=tracker, confianca=conf_min)
            alertas_prox = analisar_interacoes([p["centroide"] for p in pessoas], distancia_minima=limiar_prox)

            queda_no_frame = False
            if executar_com:
                frame_com, eventos_com = desenhar_anotacoes(
                    frame, pessoas, alertas_prox,
                    modo_velocidade=True, limiar_queda=limiar_vel, limiar_ar=limiar_ar
                )
                quedas_com += registrar_eventos(eventos_com, agrupador_com, numero_frame,
                                                "Com Velocidade", origem_rotulo, modelo_selecionado)
                queda_no_frame = queda_no_frame or any(e["tipo"] == "QUEDA" for e in eventos_com)
            if executar_sem:
                frame_sem, eventos_sem = desenhar_anotacoes(
                    frame, pessoas, alertas_prox,
                    modo_velocidade=False, limiar_queda=limiar_vel, limiar_ar=limiar_ar
                )
                quedas_sem += registrar_eventos(eventos_sem, agrupador_sem, numero_frame,
                                                "Sem Velocidade", origem_rotulo, modelo_selecionado)
                if modo_execucao == MODO_SEM:
                    queda_no_frame = any(e["tipo"] == "QUEDA" for e in eventos_sem)

            # Renderização
            if modo_execucao == MODO_AB:
                view_left.image(cv2.cvtColor(frame_sem, cv2.COLOR_BGR2RGB),
                                caption="Modelo Sem Velocidade (Muitos Falsos Positivos)", width="stretch")
                view_right.image(cv2.cvtColor(frame_com, cv2.COLOR_BGR2RGB),
                                 caption="Modelo Com Velocidade (Filtro Cinemático Ativo)", width="stretch")
            elif modo_execucao == MODO_SEM:
                view_main.image(cv2.cvtColor(frame_sem, cv2.COLOR_BGR2RGB),
                                caption=f"Modo Estático: Frame {numero_frame}", width="stretch")
            else:
                view_main.image(cv2.cvtColor(frame_com, cv2.COLOR_BGR2RGB),
                                caption=f"Modo Bimodal: Frame {numero_frame}", width="stretch")

            # Telemetria (FPS de ponta a ponta: leitura, inferência, heurísticas e exibição)
            fps_instantaneo = 1.0 / max(0.001, (time.time() - t_frame_start))
            m_fps.metric("FPS", f"{fps_instantaneo:.1f}")
            m_pessoas.metric("Pessoas em Cena", len(pessoas))
            if modo_execucao == MODO_AB:
                m_quedas.metric("Quedas (Com / Sem Vel.)", f"{quedas_com} / {quedas_sem}")
            else:
                m_quedas.metric("Quedas Registradas", quedas_com if executar_com else quedas_sem)

            if queda_no_frame:
                alert_placeholder.error(f"**ALERTA CRÍTICO DE QUEDA DETECTADO NO FRAME {numero_frame}!**")

    finally:
        cap.release()
        tempo_total = time.time() - inicio_tempo
        status_placeholder.success(
            f"Processamento concluído: {numero_frame} frames analisados em {tempo_total:.1f}s "
            f"(FPS Médio: {numero_frame / max(0.001, tempo_total):.1f})."
        )
