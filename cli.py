"""
Execução em linha de comando do pipeline de detecção de quedas (mesmo código da interface Streamlit).

Exemplos:
    python cli.py --SOURCE videos/queda_01.mp4 --MODEL yolo11s.pt --MODO com
    python cli.py --SOURCE 0 --MODEL yolo11n.pt --MODO sem
"""
import argparse
import csv
import os
from datetime import datetime

import cv2

from core.eventos import AgrupadorEventos, chave_evento
from core.heuristics import analisar_interacoes
from core.tracker import PersonTracker
from core.video_processor import desenhar_anotacoes, redimensionar_frame
from core.yolo_engine import carregar_modelo, processar_frame_detector

COLUNAS_CSV = ["Frame", "Data_Hora", "Tipo_Evento", "Detalhes", "Postura_Inferida", "Velocidade"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Sistema de Monitoramento - Detecção de Quedas (TCC UFPI)")
    parser.add_argument("--SOURCE", required=True, help="Caminho do vídeo ou 0 para webcam")
    parser.add_argument("--MODEL", default="yolo11s.pt", help="Arquivo do modelo YOLO (procura em modelos/ e na raiz)")
    parser.add_argument("--MODO", choices=["com", "sem"], default="com",
                        help="'com' = postura + velocidade (proposta); 'sem' = apenas postura (linha de base)")
    parser.add_argument("--CONF", type=float, default=0.5, help="Confiança mínima do YOLO")
    parser.add_argument("--LIMIAR-AR", type=float, default=0.75, help="Razão de aspecto abaixo da qual a pessoa está deitada")
    parser.add_argument("--LIMIAR-VEL", type=float, default=15.0, help="Velocidade mínima (px/f) para confirmar queda")
    parser.add_argument("--DIST-PROX", type=float, default=60.0, help="Distância (px) para alerta de proximidade")
    parser.add_argument("--LARGURA", type=int, default=1080, help="Largura de processamento em pixels (metodologia: 1080)")
    parser.add_argument("--SEM-CORRECAO-FPS", action="store_true",
                        help="Não converte a velocidade para a referência de 30 FPS")
    parser.add_argument("--POR-FRAME", action="store_true",
                        help="Registra todo frame que satisfaz a heurística, sem agrupar em eventos")
    parser.add_argument("--CARENCIA", type=float, default=2.0,
                        help="Segundos sem ocorrência para que a mesma pessoa/dupla gere um novo evento")
    parser.add_argument("--SEM-JANELA", action="store_true", help="Não abre a janela de visualização")
    return parser.parse_args(argv)


def caminho_relatorio(nome_fonte: str, nome_modelo: str, modo: str) -> str:
    nome_vid = os.path.splitext(nome_fonte)[0]
    nome_mod = os.path.splitext(nome_modelo)[0]
    pasta = os.path.join("relatorios", nome_vid, nome_mod)
    os.makedirs(pasta, exist_ok=True)

    base = os.path.join(pasta, f"relatorio_{nome_vid}_{nome_mod}_{modo}_velocidade")
    caminho = f"{base}.csv"
    contador = 1
    while os.path.exists(caminho):
        caminho = f"{base}_{contador}.csv"
        contador += 1
    return caminho


def main(argv=None):
    args = parse_args(argv)
    modo_velocidade = args.MODO == "com"

    fonte = 0 if args.SOURCE == "0" else args.SOURCE
    nome_fonte = "webcam" if fonte == 0 else os.path.basename(args.SOURCE)

    cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        print(f"[ERRO] Falha ao abrir o vídeo/fonte: {args.SOURCE}")
        return 1

    fps_video = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Carregando modelo {args.MODEL}...")
    modelo = carregar_modelo(args.MODEL)
    tracker = PersonTracker(fps_video=None if args.SEM_CORRECAO_FPS else fps_video)
    agrupador = AgrupadorEventos.a_partir_do_fps(fps_video, args.CARENCIA)

    caminho_csv = caminho_relatorio(nome_fonte, os.path.basename(args.MODEL), args.MODO)
    print("[INFO] Sistema iniciado. Pressione 'q' na janela do vídeo para sair.")

    numero_frame = 0
    total_quedas = 0
    with open(caminho_csv, mode="w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow(COLUNAS_CSV)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                numero_frame += 1

                frame = redimensionar_frame(frame, largura_alvo=args.LARGURA)
                pessoas = processar_frame_detector(modelo, frame, tracker=tracker, confianca=args.CONF)
                interacoes = analisar_interacoes([p["centroide"] for p in pessoas], distancia_minima=args.DIST_PROX)
                frame_anotado, eventos = desenhar_anotacoes(
                    frame, pessoas, interacoes,
                    modo_velocidade=modo_velocidade, limiar_queda=args.LIMIAR_VEL, limiar_ar=args.LIMIAR_AR
                )

                agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                for ev in eventos:
                    chave = chave_evento(ev)
                    if not args.POR_FRAME and chave is not None and not agrupador.novo_evento(chave, numero_frame):
                        continue
                    if ev["tipo"] == "QUEDA":
                        total_quedas += 1
                    escritor.writerow([numero_frame, agora, ev["tipo"], ev["detalhes"], ev["postura"], ev["velocidade"]])

                if not args.SEM_JANELA:
                    cv2.putText(frame_anotado, f"Pessoas: {len(pessoas)}", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow("Sistema de Seguranca", frame_anotado)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            if not args.SEM_JANELA:
                cv2.destroyAllWindows()

    print(f"[INFO] {numero_frame} frames processados, {total_quedas} quedas registradas. Relatório: {caminho_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
