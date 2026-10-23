import streamlit as st
import cv2
import tempfile
import time
import os
from core.yolo_engine import carregar_modelo, processar_frame_detector, MODELOS_TCC
from core.tracker import PersonTracker
from core.heuristics import analisar_interacoes
from core.video_processor import redimensionar_frame, desenhar_anotacoes
from components.ui_cards import render_alert_banner

@st.cache_resource
def get_cached_model(model_name: str):
    """Carrega o modelo com cache para evitar reinicializações desnecessárias."""
    return carregar_modelo(model_name)

def render_estudio_inferencia():
    st.title("🎥 Estúdio de Inferência e Detecção")
    st.caption("Bancada de testes interativa para execução dos modelos YOLOv11 com análise cinemática e postural")

    # Barra lateral de configurações
    st.sidebar.header("⚙️ Parâmetros de Inferência")
    
    modelo_selecionado = st.sidebar.selectbox(
        "Modelo YOLOv11",
        list(MODELOS_TCC.keys()),
        index=1,  # Default YOLOv11s (o sweet spot apontado no artigo)
        help="Selecione a variante da arquitetura YOLOv11."
    )

    modo_execucao = st.sidebar.radio(
        "Modo de Análise",
        [
            "Bimodal (Com Velocidade - Proposta)",
            "Estático (Sem Velocidade - Linha de Base)",
            "Comparativo Lado a Lado (A/B: Sem vs Com Velocidade)"
        ],
        index=0,
        help="Compare a filtragem de falsos positivos da heurística cinemática contra o modelo estático."
    )

    with st.sidebar.expander("🎛️ Calibração de Limiares Heurísticos", expanded=False):
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
            value=720,
            help="1080p é o padrão da metodologia do TCC; 720p ou 480p aceleram a execução em CPU."
        )

    # Seleção da fonte de vídeo
    col_fonte1, col_fonte2 = st.columns([2, 1])
    with col_fonte1:
        tipo_fonte = st.radio(
            "Fonte de Entrada",
            ["Upload de Arquivo de Vídeo", "Amostra Pré-Carregada", "Webcam Local"],
            horizontal=True
        )

    caminho_video = None
    origem_rotulo = "Vídeo"

    if tipo_fonte == "Upload de Arquivo de Vídeo":
        arquivo_upload = st.file_uploader("Selecione um vídeo (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        if arquivo_upload is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(arquivo_upload.read())
            caminho_video = tfile.name
            origem_rotulo = arquivo_upload.name

    elif tipo_fonte == "Amostra Pré-Carregada":
        pasta_videos = "videos"
        os.makedirs(pasta_videos, exist_ok=True)
        amostras = [f for f in os.listdir(pasta_videos) if f.lower().endswith((".mp4", ".avi", ".mov"))]
        if amostras:
            video_escolhido = st.selectbox("Escolha uma amostra disponível:", amostras)
            caminho_video = os.path.join(pasta_videos, video_escolhido)
            origem_rotulo = video_escolhido
        else:
            st.warning("Nenhum vídeo encontrado na pasta `videos/`. Faça o upload de um arquivo ou use a Webcam.")

    elif tipo_fonte == "Webcam Local":
        caminho_video = 0
        origem_rotulo = "Webcam"

    # Inicialização do histórico de eventos na sessão
    if "historico_eventos" not in st.session_state:
        st.session_state.historico_eventos = []

    st.divider()

    # Controles de Execução
    col_btn1, col_btn2, col_stats = st.columns([1, 1, 2])
    with col_btn1:
        btn_iniciar = st.button("▶️ Iniciar Análise", type="primary", use_container_width=True)
    with col_btn2:
        btn_limpar_logs = st.button("🗑️ Limpar Ocorrências", use_container_width=True)
        if btn_limpar_logs:
            st.session_state.historico_eventos = []
            st.rerun()

    # Painel de exibição
    if btn_iniciar:
        if caminho_video is None:
            st.error("Por favor, selecione uma fonte de vídeo válida antes de iniciar.")
            return

        with st.spinner("Inicializando modelo e preparando pipeline..."):
            modelo = get_cached_model(modelo_selecionado)

        tracker_a = PersonTracker()
        tracker_b = PersonTracker() if "Comparativo" in modo_execucao else None

        cap = cv2.VideoCapture(caminho_video)
        if not cap.isOpened():
            st.error("Falha ao abrir a fonte de vídeo.")
            return

        # Placeholders de visualização
        status_placeholder = st.empty()
        telemetria_col1, telemetria_col2, telemetria_col3 = st.columns(3)
        m_fps = telemetria_col1.empty()
        m_pessoas = telemetria_col2.empty()
        m_quedas = telemetria_col3.empty()

        if "Comparativo" in modo_execucao:
            col_v1, col_v2 = st.columns(2)
            view_left = col_v1.empty()
            view_right = col_v2.empty()
        else:
            view_main = st.empty()

        alert_placeholder = st.empty()

        numero_frame = 0
        total_alertas_queda = 0
        inicio_tempo = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                numero_frame += 1
                t_frame_start = time.time()

                frame = redimensionar_frame(frame, largura_alvo=resolucao_alvo)

                # Processamento com velocidade
                dados_pessoas_vel = processar_frame_detector(
                    modelo, frame, tracker=tracker_a, confianca=conf_min, usar_tracking=True
                )
                centros_a = [p["centroide"] for p in dados_pessoas_vel]
                alertas_prox_a = analisar_interacoes(centros_a, distancia_minima=limiar_prox)

                frame_anotado_vel, eventos_vel = desenhar_anotacoes(
                    frame, dados_pessoas_vel, alertas_prox_a,
                    modo_velocidade=True, limiar_queda=limiar_vel, limiar_ar=limiar_ar
                )

                # Registro de eventos na sessão
                for ev in eventos_vel:
                    if ev["tipo"] == "QUEDA":
                        total_alertas_queda += 1
                        st.session_state.historico_eventos.append({
                            "Frame": numero_frame,
                            "Timestamp": time.strftime("%H:%M:%S"),
                            "Tipo": ev["tipo"],
                            "Postura": ev["postura"],
                            "Velocidade": ev["velocidade"],
                            "Detalhes": ev["detalhes"],
                            "Fonte": origem_rotulo,
                            "Modelo": modelo_selecionado
                        })

                # Cálculo de FPS
                fps_instantaneo = 1.0 / max(0.001, (time.time() - t_frame_start))

                # Atualiza telemetria
                m_fps.metric("⚡ FPS", f"{fps_instantaneo:.1f}")
                m_pessoas.metric("👥 Pessoas em Cena", len(dados_pessoas_vel))
                m_quedas.metric("🚨 Quedas Registradas", total_alertas_queda)

                if eventos_vel and any(e["tipo"] == "QUEDA" for e in eventos_vel):
                    alert_placeholder.error(f"🚨 **ALERTA CRÍTICO DE QUEDA DETECTADO NO FRAME {numero_frame}!**")

                # Renderização
                if "Comparativo" in modo_execucao:
                    # Ramo Sem Velocidade
                    dados_pessoas_sem_vel = processar_frame_detector(
                        modelo, frame, tracker=None, confianca=conf_min, usar_tracking=False
                    )
                    centros_b = [p["centroide"] for p in dados_pessoas_sem_vel]
                    alertas_prox_b = analisar_interacoes(centros_b, distancia_minima=limiar_prox)
                    frame_anotado_sem_vel, _ = desenhar_anotacoes(
                        frame, dados_pessoas_sem_vel, alertas_prox_b,
                        modo_velocidade=False, limiar_queda=limiar_vel, limiar_ar=limiar_ar
                    )

                    rgb_sem_vel = cv2.cvtColor(frame_anotado_sem_vel, cv2.COLOR_BGR2RGB)
                    rgb_com_vel = cv2.cvtColor(frame_anotado_vel, cv2.COLOR_BGR2RGB)

                    view_left.image(rgb_sem_vel, caption="❌ Modelo Sem Velocidade (Muitos Falsos Positivos)", use_container_width=True)
                    view_right.image(rgb_com_vel, caption="✅ Modelo Com Velocidade (Filtro Cinemático Ativo)", use_container_width=True)

                elif "Sem Velocidade" in modo_execucao:
                    dados_pessoas_sem_vel = processar_frame_detector(
                        modelo, frame, tracker=None, confianca=conf_min, usar_tracking=False
                    )
                    centros_b = [p["centroide"] for p in dados_pessoas_sem_vel]
                    alertas_prox_b = analisar_interacoes(centros_b, distancia_minima=limiar_prox)
                    frame_anotado_sem_vel, _ = desenhar_anotacoes(
                        frame, dados_pessoas_sem_vel, alertas_prox_b,
                        modo_velocidade=False, limiar_queda=limiar_vel, limiar_ar=limiar_ar
                    )
                    rgb_sem_vel = cv2.cvtColor(frame_anotado_sem_vel, cv2.COLOR_BGR2RGB)
                    view_main.image(rgb_sem_vel, caption=f"Modo Estático: Frame {numero_frame}", use_container_width=True)

                else:
                    rgb_com_vel = cv2.cvtColor(frame_anotado_vel, cv2.COLOR_BGR2RGB)
                    view_main.image(rgb_com_vel, caption=f"Modo Bimodal: Frame {numero_frame}", use_container_width=True)

        finally:
            cap.release()
            tempo_total = time.time() - inicio_tempo
            status_placeholder.success(f"Processamento concluído: {numero_frame} frames analisados em {tempo_total:.1f}s (FPS Médio: {numero_frame / max(1.0, tempo_total):.1f}).")
