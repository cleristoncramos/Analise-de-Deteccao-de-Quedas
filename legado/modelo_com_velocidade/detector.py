import os
import math
from ultralytics import YOLO

# def inicializar_modelo(caminho_pesos='modelos\\yolo11n.pt', dispositivo='cuda:0'):
#     print("[INFO] Carregando modelo YOLO...")
#     return YOLO(caminho_pesos).to(dispositivo)

def inicializar_modelo(caminho_pesos=os.path.join('modelos', 'yolo11n.pt')):
    print("[INFO] Carregando modelo YOLO...")
    return YOLO(caminho_pesos)

def detectar_pessoas(modelo, frame, confianca=0.5):
    resultado = modelo(frame, conf=confianca, verbose=False)[0]
    boxes = []
    centros = []

    for r in resultado.boxes:
        cls = int(r.cls[0])
        if cls == 0:
            x1, y1, x2, y2 = r.xyxy[0].tolist()
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            boxes.append([x1, y1, x2, y2])
            centros.append((cx, cy))
    
    return boxes, centros

posicoes_anteriores = {}

def detectar_pessoas_com_velocidade(modelo, frame, confianca=0.5):
    global posicoes_anteriores

    resultado = modelo.track(frame, persist=True, conf=confianca, verbose=False)[0]

    boxes = []
    centros = []
    velocidades = []

    if resultado.boxes.id is not None:
        ids_rastreados = resultado.boxes.id.int().tolist()

        for r, obj_id in zip(resultado.boxes, ids_rastreados):
            cls = int(r.cls[0])
            if cls == 0:
                x1, y1, x2, y2 = r.xyxy[0].int().tolist()

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                velocidade_atual = 0.0
                if obj_id in posicoes_anteriores:
                    cx_ant, cy_ant = posicoes_anteriores[obj_id]
                    velocidade_atual = math.sqrt((cx - cx_ant)**2 + (cy - cy_ant)**2)
                
                posicoes_anteriores[obj_id] = (cx, cy)

                boxes.append([x1, y1, x2, y2])
                centros.append((cx, cy))
                velocidades.append(velocidade_atual)

    return boxes, centros, velocidades