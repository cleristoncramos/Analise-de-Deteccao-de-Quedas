import os
import torch
from typing import List, Dict, Any, Optional
from ultralytics import YOLO
from .tracker import PersonTracker

# Mapeamento dos modelos pré-treinados avaliados no TCC
MODELOS_TCC = {
    "YOLOv11 Nano (v11n)": "yolo11n.pt",
    "YOLOv11 Small (v11s)": "yolo11s.pt",
    "YOLOv11 Medium (v11m)": "yolo11m.pt",
    "YOLOv11 Large (v11l)": "yolo11l.pt",
    "YOLOv11 Extra Large (v11x)": "yolo11x.pt"
}

def obter_caminho_modelo(nome_ou_arquivo: str) -> str:
    """
    Localiza o arquivo .pt na pasta 'modelos/' ou na raiz, ou retorna o identificador Ultralytics.
    """
    arquivo_alvo = MODELOS_TCC.get(nome_ou_arquivo, nome_ou_arquivo)
    
    # 1. Procura na pasta local 'modelos/'
    caminho_local_modelos = os.path.join("modelos", arquivo_alvo)
    if os.path.isfile(caminho_local_modelos):
        return caminho_local_modelos
        
    # 2. Procura na raiz do projeto
    if os.path.isfile(arquivo_alvo):
        return arquivo_alvo

    # 3. Retorna o nome base (o Ultralytics fará o download automático dos pesos oficiais)
    return arquivo_alvo

def carregar_modelo(caminho_ou_nome: str, dispositivo: Optional[str] = None) -> YOLO:
    """
    Carrega o modelo YOLO na memória com opção de acelerador CPU ou GPU.
    """
    caminho = obter_caminho_modelo(caminho_ou_nome)

    if dispositivo is None:
        dispositivo = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    modelo = YOLO(caminho)
    # ultralytics permite transferir dispositivo
    if dispositivo != "cpu" and torch.cuda.is_available():
        modelo.to(dispositivo)
        
    return modelo

def reiniciar_rastreamento(modelo: YOLO) -> None:
    """
    Zera o estado do rastreador interno do Ultralytics (IDs e filtros de Kalman).
    Com `track(persist=True)` esse estado sobrevive entre chamadas, então deve
    ser zerado antes de começar um vídeo novo.
    """
    predictor = getattr(modelo, "predictor", None)
    for rastreador in getattr(predictor, "trackers", None) or []:
        rastreador.reset()

def processar_frame_detector(
    modelo: YOLO,
    frame,
    tracker: Optional[PersonTracker] = None,
    confianca: float = 0.5,
    usar_tracking: bool = True
) -> List[Dict[str, Any]]:
    """
    Executa a inferência YOLO em um frame e filtra a classe 0 (pessoa).
    Se usar_tracking for True e um tracker for fornecido, calcula a velocidade e trajetórias.
    """
    if usar_tracking and tracker is not None:
        resultado = modelo.track(frame, persist=True, conf=confianca, verbose=False)[0]
    else:
        resultado = modelo(frame, conf=confianca, verbose=False)[0]

    boxes_pessoas = []
    track_ids = []

    if resultado.boxes is not None and len(resultado.boxes) > 0:
        for r in resultado.boxes:
            cls = int(r.cls[0])
            if cls == 0:  # Classe Pessoa (COCO)
                boxes_pessoas.append(r.xyxy[0].cpu().numpy().tolist())
                # Sem ID nativo do rastreador, a pessoa fica sem identidade (None)
                # em vez de receber um número que pode colidir com outro ID real
                track_ids.append(int(r.id[0]) if r.id is not None else None)

    if tracker is not None and usar_tracking:
        return tracker.atualizar(boxes_pessoas, track_ids)

    # Modo sem tracking: retorna apenas centroides estáticos
    resultados = []
    for box in boxes_pessoas:
        x1, y1, x2, y2 = box
        largura = max(0.0, float(x2 - x1))
        altura = max(0.0, float(y2 - y1))
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        resultados.append({
            "id": None,
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "largura": largura,
            "altura": altura,
            "centroide": (cx, cy),
            "velocidade": 0.0,
            "trajetoria": [(cx, cy)]
        })

    return resultados
