import argparse
import cv2
import os
import logger
import heuristica_risco
import detector
import analisador_movimento

def parse_args():
    parser = argparse.ArgumentParser(description="Sistema de Monitoramento")
    parser.add_argument('--SOURCE', type=str, default=os.path.join('videos', 'pessoas_andando.mp4'), help='Caminho do video ou 0 para webcam')
    parser.add_argument('--MODEL', type=str, default=os.path.join('modelos', 'yolo11n.pt'), help='Caminho do modelo YOLO')
    return parser.parse_args()

def main():
    args = parse_args()

    modelo = detector.inicializar_modelo(args.MODEL)
    
    source = int(args.SOURCE) if args.SOURCE == '0' else args.SOURCE
    if isinstance(source, str):
        nome_video = os.path.basename(source)
        nome_modelo = os.path.basename(args.MODEL)
        arquivo_log, escritor_log = logger.iniciar_log(nome_video, nome_modelo)
    else:
        arquivo_log, escritor_log = logger.iniciar_log('webcam', os.path.basename(args.MODEL))

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERRO] Falha ao abrir o vídeo/fonte: {args.SOURCE}")
        return
    
    print("[INFO] Sistema Iniciando. Pressione 'q' na janela do vídeo para sair.")

    numero_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        numero_frame += 1

        LARGURA_PADRAO = 1080
        altura_original, largura_original =  frame.shape[:2]
        proporcao = LARGURA_PADRAO / largura_original
        altura_calculada = int(altura_original * proporcao)

        frame = cv2.resize(frame, (LARGURA_PADRAO, altura_calculada))

        frame_proc = frame.copy()

        boxes, centros, velocidades = detector.detectar_pessoas_com_velocidade(modelo, frame_proc)

        posturas_do_frame = []
        acoes_do_frame = []

        for box, velocidade in zip(boxes, velocidades):
            x1, y1, x2, y2 = box
            largura = x2 - x1
            altura = y2 - y1

            postura = heuristica_risco.analisar_postura(largura, altura)
            posturas_do_frame.append(postura)
            
            valor_vel = velocidade[0] if isinstance(velocidade, list) else velocidade
            vel_formatada = f"{float(valor_vel):.2f} px/f"
            
            acao = analisador_movimento.classificar_acao(valor_vel)
            acoes_do_frame.append(acao)

            alerta_queda = analisador_movimento.confirmar_queda(postura, valor_vel)

            cor = (255, 0, 0)
            
            if alerta_queda:
                cor = (0, 0, 255)
                texto_tela = "ALERTA: QUEDA"
                logger.registrar_evento(escritor_log, numero_frame, "QUEDA", f"Deteccao de queda ({acao})", postura, vel_formatada)
            else:
                cor = (255, 0, 0)

                texto_tela = f"{vel_formatada} | {postura} | ({acao})"
            
            cv2.rectangle(frame_proc, (int(x1), int(y1)), (int(x2), int(y2)), cor, 2)
            cv2.putText(frame_proc, texto_tela, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cor, 2)

        # Analisar proximidade
        alerta_proximidade = heuristica_risco.analisar_interacoes(centros)
        
        for (i, j, p1, p2, dist) in alerta_proximidade:
            cv2.line(frame_proc, p1, p2, (0, 0, 255), 2)
            cv2.circle(frame_proc, (int((p1[0]+p2[0])/2), int((p1[1]+p2[1])/2)), 5, (0,0,255), -1)
            
            postura_envolvidos = f"{posturas_do_frame[i]} e {posturas_do_frame[j]}"
            acao_envolvidos = f"{acoes_do_frame[i]} e {acoes_do_frame[j]}"
            
            logger.registrar_evento(escritor_log, numero_frame, "PROXIMIDADE", f"Distancia: {dist:.2f}px", postura_envolvidos, acao_envolvidos)

        cv2.putText(frame_proc, f"Pessoas: {len(boxes)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Sistema de Seguranca", frame_proc)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    logger.fechar_log(arquivo_log)
    cv2.destroyAllWindows()
    print("[INFO] Processamento finalizado.")

if __name__ == '__main__':
    main()