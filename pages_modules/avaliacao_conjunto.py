"""Avaliação do conjunto do UR Fall: percorre os vídeos e recalcula as métricas a cada vídeo concluído.

Dois modos:
- Resultados pré-calculados: reproduz, vídeo a vídeo, a avaliação feita pelo CLI com os parâmetros
  padrão (data/urfall_resultados.csv). É instantâneo e serve para a apresentação.
- Rodar de verdade: analisa cada vídeo com o modelo e os limiares da barra lateral. Cada vídeo
  concluído fica no cache de resultados, então uma avaliação interrompida continua de onde parou.
"""
import csv
import io
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from core.ambiente import padroes_execucao
from core.analise_video import analisar_video
from core.cache_resultados import carregar_resultado, chave_resultado, salvar_resultado
from core.catalogo_videos import ROTULOS, ErroDownloadVideo, carregar_catalogo, obter_video
from core.metricas_conjunto import (
    ADL_DO_ARTIGO,
    carregar_pre_calculados,
    classificar,
    linha_do_artigo,
    metricas,
    parametros_compativeis,
    selecionar_videos,
)
from core.yolo_engine import MODELOS_TCC
from pages_modules.estudio_inferencia import VERSAO_ANALISE, _parametros_barra_lateral, obter_modelo_da_sessao

ORIGEM_PRE = "Resultados pré-calculados"
ORIGEM_AO_VIVO = "Rodar de verdade"
CONJUNTO_ARTIGO = "30 primeiras ADL, como no artigo (60 vídeos)"
CONJUNTO_TODO = "Todas as 40 ADL (70 vídeos)"
LIMITE_TODOS = "Todos os vídeos do conjunto"
LIMITE_QUEDAS = "Só as 30 quedas"
LIMITE_N = "Só os primeiros N vídeos"
PAUSA_ANIMACAO_S = 0.12
# Segundos por quadro em CPU a 1080 px (medidos na avaliação pelo CLI), só para a estimativa antes de começar
SEGUNDOS_POR_QUADRO = {"n": 0.2, "s": 0.4, "m": 1.0, "l": 2.0, "x": 2.4}
NOME_MODO = {"com": "Com velocidade", "sem": "Sem velocidade"}
CLASSE_CSS = {"VP": "acerto", "VN": "acerto", "FP": "erro", "FN": "erro"}


def _fmt(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def tabela_metricas(linhas, arquivo_modelo: str, n_adl: int) -> pd.DataFrame:
    """Métricas dos vídeos já avaliados nos dois modos, com a linha da Tabela 2 do artigo logo abaixo."""
    registros = []
    for modo in ("com", "sem"):
        m = metricas(linha[f"classe_{modo}"] for linha in linhas)
        registros.append({"Cenário": NOME_MODO[modo], "Origem": "Esta avaliação", "Vídeos": m["total"],
                          "TN": m["tn"], "FP": m["fp"], "FN": m["fn"], "TP": m["tp"], "Acurácia": _fmt(m["acuracia"]),
                          "Precisão": _fmt(m["precisao"]), "Revocação": _fmt(m["recall"]),
                          "F1": _fmt(m["f1_score"])})
        artigo = linha_do_artigo(arquivo_modelo, modo)
        if artigo:
            registros.append({"Cenário": NOME_MODO[modo], "Origem": "Artigo (Tabela 2)", "Vídeos": 2 * ADL_DO_ARTIGO,
                              "TN": artigo["tn"], "FP": artigo["fp"], "FN": artigo["fn"], "TP": artigo["tp"],
                              "Acurácia": _fmt(artigo["acuracia"]), "Precisão": _fmt(artigo["precisao"]),
                              "Revocação": _fmt(artigo["recall"]), "F1": _fmt(artigo["f1_score"])})
    return pd.DataFrame(registros)


def tabela_videos(linhas) -> pd.DataFrame:
    return pd.DataFrame([{
        "Vídeo": linha["video"], "Esperado": ROTULOS[linha["rotulo"]],
        "Quedas (com vel.)": linha["quedas_com"], "Com velocidade": linha["classe_com"],
        "Quedas (sem vel.)": linha["quedas_sem"], "Sem velocidade": linha["classe_sem"],
        "Origem": linha["origem"],
    } for linha in linhas][::-1])  # o vídeo mais recente no topo


def _colorir(valor):
    if valor in ("VP", "VN"):
        return "color: #4f6d5a; font-weight: 600"
    if valor in ("FP", "FN"):
        return "color: #7b2d26; font-weight: 600"
    return ""


def csv_resultados(linhas, cabecalho: dict) -> bytes:
    saida = io.StringIO()
    for chave, valor in cabecalho.items():
        saida.write(f"# {chave}: {valor}\n")
    escritor = csv.DictWriter(saida, fieldnames=["video", "rotulo", "quedas_com", "classe_com", "quedas_sem",
                                                 "classe_sem", "origem"], extrasaction="ignore")
    escritor.writeheader()
    escritor.writerows(linhas)
    return saida.getvalue().encode("utf-8-sig")


def _desenhar(areas, linhas, arquivo_modelo, n_adl, chave):
    areas["metricas"].dataframe(tabela_metricas(linhas, arquivo_modelo, n_adl), hide_index=True, width="stretch",
                                key=f"metricas_{chave}")
    if linhas:
        areas["videos"].dataframe(tabela_videos(linhas).style.map(_colorir, subset=["Com velocidade", "Sem velocidade"]),
                                  hide_index=True, width="stretch", height=min(420, 35 * (len(linhas) + 1) + 3),
                                  key=f"videos_{chave}")


def _linha(video, quedas_com, quedas_sem, origem):
    return {"video": video["id"], "rotulo": video["rotulo"], "quedas_com": quedas_com,
            "classe_com": classificar(video["rotulo"], quedas_com), "quedas_sem": quedas_sem,
            "classe_sem": classificar(video["rotulo"], quedas_sem), "origem": origem}


def render_avaliacao_conjunto():
    st.markdown('<div class="estudio-cabecalho"><span class="titulo">Avaliação do conjunto</span>'
                '<span class="sub">UR Fall Detection Dataset · câmera 0 · imagem colorida · avaliação por vídeo</span>'
                '</div>', unsafe_allow_html=True)

    padroes = padroes_execucao()
    (modelo_selecionado, _modo, conf_min, limiar_ar, limiar_vel, limiar_prox, resolucao_alvo,
     corrigir_fps, carencia_s, pular_quadros) = _parametros_barra_lateral(padroes)
    arquivo_modelo = MODELOS_TCC[modelo_selecionado]
    chave_modelo = Path(arquivo_modelo).stem
    compativel = parametros_compativeis(conf=conf_min, ar=limiar_ar, vel=limiar_vel, largura=resolucao_alvo,
                                        corrigir_fps=corrigir_fps, pular=pular_quadros)

    try:
        catalogo = carregar_catalogo()
    except (OSError, ValueError) as exc:
        st.error(f"Não foi possível ler o catálogo de vídeos ({exc}).")
        return

    col_conjunto, col_origem, col_acao = st.columns([1, 1, 1], gap="large")
    with col_conjunto:
        conjunto = st.radio("Conjunto avaliado", [CONJUNTO_ARTIGO, CONJUNTO_TODO], key="conjunto_adl")
        n_adl = ADL_DO_ARTIGO if conjunto == CONJUNTO_ARTIGO else 40
    with col_origem:
        if not compativel:
            st.session_state["conjunto_origem"] = ORIGEM_AO_VIVO  # o pré-calculado não vale para outros parâmetros
        origem = st.radio("Origem dos resultados", [ORIGEM_PRE, ORIGEM_AO_VIVO], key="conjunto_origem",
                          disabled=not compativel,
                          help="Pré-calculados: reproduz, vídeo a vídeo, a avaliação feita pelo CLI com os parâmetros "
                               "padrão. Rodar de verdade: analisa cada vídeo agora, com o modelo e os limiares da "
                               "barra lateral.")
        if not compativel:
            st.warning("Os parâmetros da barra lateral foram alterados. Os resultados pré-calculados só valem para os "
                       "parâmetros padrão (confiança 0,5, AR 0,75, velocidade 15 px/f, 1080 px, FPS normalizado e "
                       "todos os quadros), então só é possível **rodar de verdade**.")

        videos = selecionar_videos(catalogo["videos"], n_adl=n_adl)
        if origem == ORIGEM_AO_VIVO:
            limite = st.selectbox("Vídeos a analisar", [LIMITE_TODOS, LIMITE_QUEDAS, LIMITE_N], key="conjunto_limite")
            if limite == LIMITE_QUEDAS:
                videos = selecionar_videos(catalogo["videos"], somente_quedas=True)
            elif limite == LIMITE_N:
                n = st.number_input("N", 1, len(videos), min(10, len(videos)), key="conjunto_n")
                videos = videos[:n]
            quadros = sum(v.get("quadros", 0) for v in videos)
            estimativa = quadros * SEGUNDOS_POR_QUADRO.get(chave_modelo[-1], 0.5) / 60
            texto = (f"{len(videos)} vídeos, {quadros} quadros: cerca de {max(1, round(estimativa))} min "
                     "(estimativa grosseira, refeita ao vivo). Vídeos já analisados com os mesmos parâmetros saem do "
                     "cache na hora.")
            if padroes.modo_leve:
                st.warning(f"{texto} Na versão publicada, a avaliação completa leva horas e pode provocar a limitação "
                           "de CPU da plataforma. Prefira os resultados pré-calculados ou limite os vídeos.")
            else:
                st.caption(texto)
            if pular_quadros > 1:
                st.caption("A avaliação do conjunto analisa todos os quadros; a opção de pular quadros não se aplica.")

    with col_acao:
        col_b1, col_b2 = st.columns([1.4, 1])
        with col_b1:
            iniciar = st.button("Avaliar todos", type="primary", icon=":material/playlist_play:", width="stretch")
        with col_b2:
            st.button("Parar", icon=":material/stop:", width="stretch", key="conjunto_parar")
        ativo = ("reprodução da avaliação pré-calculada (CLI, PyTorch, parâmetros padrão)" if origem == ORIGEM_PRE
                 else "análise ao vivo com o modelo e os limiares da barra lateral")
        st.markdown(f'<div class="conjunto-modo"><b>Modo ativo:</b> {ativo}</div>', unsafe_allow_html=True)
        area_status = st.empty()
        barra = st.empty()
        area_download = st.empty()

    st.markdown(f"**Métricas recalculadas a cada vídeo · {modelo_selecionado}**")
    areas = {"metricas": st.empty()}
    st.caption("Cada vídeo é um evento: queda com ao menos uma queda registrada = VP (senão FN); atividade diária com "
               "queda registrada = FP (senão VN). Os dois modos são avaliados juntos, com uma única inferência por "
               "quadro. A Tabela 2 do artigo usa as 30 primeiras atividades diárias (60 vídeos).")
    st.markdown("**Resultado por vídeo** (o mais recente no topo)")
    areas["videos"] = st.empty()

    config = {"modelo": arquivo_modelo, "origem": origem, "conjunto": conjunto, "conf": conf_min, "ar": limiar_ar,
              "vel": limiar_vel, "largura": resolucao_alvo, "corrigir_fps": corrigir_fps}
    anterior = st.session_state.get("conjunto_resultado")

    if not iniciar:
        if anterior and anterior["config"] == config:
            _desenhar(areas, anterior["linhas"], arquivo_modelo, n_adl, "salvo")
            situacao = "concluída" if anterior["concluido"] else "interrompida"
            area_status.caption(f"Última avaliação {situacao}: {len(anterior['linhas'])} de {anterior['total']} vídeos.")
            _botao_download(area_download, anterior["linhas"], config)
        else:
            _desenhar(areas, [], arquivo_modelo, n_adl, "vazio")
            area_status.caption("Clique em **Avaliar todos** para percorrer os vídeos.")
        return

    linhas = []
    estado = {"config": config, "linhas": linhas, "total": len(videos), "concluido": False}
    st.session_state.conjunto_resultado = estado
    _desenhar(areas, linhas, arquivo_modelo, n_adl, "inicio")

    if origem == ORIGEM_PRE:
        pre = carregar_pre_calculados().get(chave_modelo, {})
        for n, video in enumerate(videos, start=1):
            if video["id"] not in pre:
                continue
            r = pre[video["id"]]
            linhas.append(_linha(video, r["com"], r["sem"], "pré-calculado"))
            barra.progress(n / len(videos), text=f"Vídeo {n} de {len(videos)} · {video['id']}")
            _desenhar(areas, linhas, arquivo_modelo, n_adl, n)
            time.sleep(PAUSA_ANIMACAO_S)
    else:
        with st.spinner("Carregando o modelo..."):
            modelo = obter_modelo_da_sessao(modelo_selecionado, padroes.usar_openvino)
        inicio = time.time()
        quadros_feitos = 0
        quadros_total = sum(v.get("quadros", 0) for v in videos) or 1
        for n, video in enumerate(videos, start=1):
            chave = chave_resultado(versao=VERSAO_ANALISE, tipo="conjunto", video=video["id"], modelo=modelo_selecionado,
                                    conf=conf_min, ar=limiar_ar, vel=limiar_vel, prox=limiar_prox,
                                    largura=resolucao_alvo, corrigir_fps=corrigir_fps, carencia=carencia_s,
                                    openvino=padroes.usar_openvino)
            guardado = carregar_resultado(chave)
            if guardado:
                dados = guardado[0]
                linhas.append(_linha(video, dados["com"], dados["sem"], "cache"))
            else:
                try:
                    caminho = obter_video(video)
                except ErroDownloadVideo as exc:
                    area_status.warning(f"{video['id']} pulado: {exc}")
                    continue
                ultimo = [0.0]
                base = quadros_feitos
                t0 = time.time()

                def ao_quadro(quadro, video=video, n=n, base=base, ultimo=ultimo):
                    agora = time.time()
                    if agora - ultimo[0] >= 0.5:
                        ultimo[0] = agora
                        feitos = base + quadro
                        ritmo = feitos / max(0.001, agora - inicio)
                        restante = max(0, quadros_total - feitos) / max(ritmo, 0.01)
                        barra.progress(min(1.0, feitos / quadros_total),
                                       text=f"Vídeo {n} de {len(videos)} · {video['id']} · quadro {quadro} · "
                                            f"cerca de {restante / 60:.0f} min restantes")
                    if padroes.pausa_s:
                        time.sleep(padroes.pausa_s)

                resultado = analisar_video(cv2.VideoCapture(str(caminho)), modelo, modos=("com", "sem"),
                                           conf=conf_min, limiar_ar=limiar_ar, limiar_vel=limiar_vel,
                                           dist_prox=limiar_prox, largura=resolucao_alvo, corrigir_fps=corrigir_fps,
                                           carencia_s=carencia_s, recorte=video.get("recorte"), ao_quadro=ao_quadro)
                dados = {"com": resultado["quedas"]["com"], "sem": resultado["quedas"]["sem"],
                         "frames": resultado["frames"], "tempo_s": time.time() - t0}
                try:
                    salvar_resultado(chave, dados)
                except OSError:
                    pass
                linhas.append(_linha(video, dados["com"], dados["sem"], "analisado agora"))
            quadros_feitos += video.get("quadros", 0)
            barra.progress(min(1.0, quadros_feitos / quadros_total), text=f"Vídeo {n} de {len(videos)} · {video['id']}")
            _desenhar(areas, linhas, arquivo_modelo, n_adl, n)

    estado["concluido"] = True
    barra.empty()
    area_status.markdown(f'<div class="estudio-status">Concluído: {len(linhas)} vídeos avaliados.</div>',
                         unsafe_allow_html=True)
    _botao_download(area_download, linhas, config)


def _botao_download(area, linhas, config):
    if linhas:
        area.download_button("Baixar resultados (CSV)", csv_resultados(linhas, config),
                             file_name=f"urfall_{Path(config['modelo']).stem}.csv", mime="text/csv",
                             icon=":material/download:", on_click="ignore", width="stretch")
