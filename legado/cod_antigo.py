import argparse
import os
import time
import cv2
import csv
import math
from datetime import datetime
from itertools import combinations
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Sistema de Monitoramento Escolar - TCC")
    parser.add_argument('--SOURCE', type=str, default=os.path.join('videos', 'pessoas_andando.mp4'), help='Caminho do vídeo ou 0 para webcam')
    parser.add_argument('--MODEL', type=str, default=os.path.join('modelos', 'yolo11x.pt'), help='Nome do arquivo do modelo YOLO')
    parser.add_argument('--THRESHOLD', type=int, default=10, help='Limite de pessoas para aglomeração')
    parser.add_argument('--ROI', type=str, default=None, help='ROI como x1,y1,x2,y2')
    return parser.parse_args()

def main():
    args = parse_args()

    nome_base = os.path.splitext(os.path.basename(args.SOURCE))[0]if args.SOURCE != '0' else "webcam"
    nome_modelo = os.path.splitext(os.path.basename(args.MODEL))[0]
    
    print(f"[INFO] Carregando modelo YOLO {nome_base}")
    model = YOLO(args.MODEL)

    nome_log = f'relatorio_ocorrencias_{nome_base}_{nome_modelo}.csv'
    arquivo_log = open(nome_log, mode='w', newline='', encoding='utf-8')
    escritor_log = csv.writer(arquivo_log)
    escritor_log.writerow(['Data_Hora', 'Tipo_Evento', 'Detalhes'])

    source = int(args.SOURCE) if args.SOURCE == '0' else args.SOURCE
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERRO] Não foi possível abrir o vídeo/fonte: {args.SOURCE}")
        print("Verifique se o nome do arquivo está correto e na mesma pasta do script.")
        return

    print("[INFO] Sistema Iniciado. Pressione 'q' na janela do vídeo para sair.")

    roi = None
    if args.ROI:
        vals = list(map(int, args.ROI.split(',')))
        roi = tuple(vals)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fim do vídeo alcançado.")
            break

        frame_proc = frame.copy()

        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0, 255, 0), 2)
            crop = frame[y1:y2, x1:x2]
            results = model(crop, conf=0.5, verbose=False)[0]
        else:
            results = model(frame_proc, conf=0.5, verbose=False)[0]

        boxes = []
        centros = []

        for r in results.boxes:
            cls = int(r.cls[0])
            if cls == 0: # Classe 0 = Pessoa
                x1, y1, x2, y2 = r.xyxy[0].int().tolist()

                if roi:
                    x1 += roi[0]; x2 += roi[0]
                    y1 += roi[1]; y2 += roi[1]

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centros.append((cx, cy))
                boxes.append([x1, y1, x2, y2])

                width = x2 - x1
                height = y2 - y1
                if width <= 0:
                    continue
                ratio = height / width

                if ratio < 0.75:
                    cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_proc, "QUEDA", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    escritor_log.writerow([agora, "QUEDA", f'Ratio: {ratio:.2f}'])
                else:
                    cv2.rectangle(frame_proc, (x1, y1), (x2, y2), (255, 0, 0), 2)

        distancia_minima = 60

        for (cx_a, cy_a), (cx_b, cy_b) in combinations(centros, 2):
            distancia = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)

            if distancia < distancia_minima:
                cv2.line(frame_proc, (cx_a, cy_a), (cx_b, cy_b), (0, 0, 255), 2)
                cv2.circle(frame_proc, (int((cx_a + cx_b) / 2), int((cy_a + cy_b) / 2)), 5, (0, 0, 255), -1)
                agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                escritor_log.writerow([agora, "PROXIMIDADE", f'Distancia: {distancia:.2f}px'])

        cv2.putText(frame_proc, f"Pessoas: {len(boxes)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Sistema de Seguranca Escolar - TCC", frame_proc)
        
        if cv2.waitKey(30) & 0xFF == ord('q'): 
            break

    cap.release()
    arquivo_log.close()
    cv2.destroyAllWindows()
    print(f"[INFO] Processamento finalizado. Logs salvos em '{nome_log}'")

if __name__ == '__main__':
    main()