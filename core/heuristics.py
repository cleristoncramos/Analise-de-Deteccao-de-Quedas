import math
from itertools import combinations
from typing import List, Tuple, Dict, Any

def analisar_postura(largura: float, altura: float, limiar_deitado: float = 0.75, limiar_sentado: float = 1.2) -> Tuple[str, float]:
    """
    Avalia a Razão de Aspecto (AR = altura / largura) de uma bounding box.
    Retorna a postura inferida e o valor do ratio.
    """
    if largura <= 0:
        return "Indefinido", 0.0
    
    ratio = altura / largura

    if ratio < limiar_deitado:
        return "Deitado", ratio
    elif limiar_deitado <= ratio <= limiar_sentado:
        return "Sentado/Agachado", ratio
    else:
        return "Em pe", ratio

def classificar_acao(velocidade: float, limite_andando: float = 1.0, limite_correndo: float = 30.0) -> str:
    """
    Segmenta a velocidade escalar em categorias cinemáticas conforme o artigo.
    """
    if velocidade < limite_andando:
        return "Mexendo-se Pouco"
    elif velocidade < limite_correndo:
        return "Andando"
    else:
        return "Correndo"

def confirmar_queda(
    postura: str,
    velocidade: float,
    limiar_queda: float = 15.0,
    usar_velocidade: bool = True
) -> bool:
    """
    Confirmação do evento de risco de queda:
    - Modo Bimodal (usar_velocidade=True): Postura == 'Deitado' AND velocidade >= limiar_queda
    - Modo Estático (usar_velocidade=False): Postura == 'Deitado'
    """
    if postura != "Deitado":
        return False
    
    if not usar_velocidade:
        return True
    
    return float(velocidade) >= float(limiar_queda)

def analisar_interacoes(centros: List[Tuple[int, int]], distancia_minima: float = 60.0) -> List[Dict[str, Any]]:
    """
    Calcula pares de indivíduos cuja distância euclidiana seja menor que distancia_minima (pixels).
    Retorna lista de dicionários com os índices, coordenadas e distância calculada.
    """
    alertas = []
    for i, j in combinations(range(len(centros)), 2):
        cx_a, cy_a = centros[i]
        cx_b, cy_b = centros[j]
        distancia = math.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)
        if distancia < distancia_minima:
            alertas.append({
                "idx_a": i,
                "idx_b": j,
                "ponto_a": (cx_a, cy_a),
                "ponto_b": (cx_b, cy_b),
                "ponto_medio": (int((cx_a + cx_b) / 2), int((cy_a + cy_b) / 2)),
                "distancia": distancia
            })
    return alertas

def calcular_velocidade_normalizada(velocidade: float, largura: float, altura: float) -> float:
    """
    Normaliza a velocidade pela escala da bounding box para atenuar a perda de escala no Eixo Z.
    v_norm = (v / sqrt(w * h)) * 100
    """
    area = largura * altura
    if area <= 0:
        return 0.0
    return (velocidade / math.sqrt(area)) * 100.0
