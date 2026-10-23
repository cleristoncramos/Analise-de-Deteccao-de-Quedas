import os
from ultralytics import YOLO

""""
Para usar em computado com placa de vídeo dedicadada 

Usar apenas quando tiver placa RTX
"""
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
            coords = r.xyxy[0].tolist()
            x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            boxes.append([x1, y1, x2, y2])
            centros.append((cx, cy))
    
    return boxes, centros