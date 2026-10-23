import math
from itertools import combinations

def analisar_postura(largura, altura):
    if largura <= 0:
        return "Indefinido", False
    
    ratio = altura / largura

    if ratio < 0.75:
        return "Deitado", True
    elif 0.75 <= ratio <= 1.2:
        return "Sentado/Agachado", False
    else:
        return "Em pe", False
    
def analisar_interacoes(centros, distancia_minima=60):
    alertas = []
    for (cx_a, cy_a), (cx_b, cy_b) in combinations(centros, 2):
        distancia = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
        if distancia < distancia_minima:
            alertas.append(((cx_a, cy_a), (cx_b, cy_b), distancia))
    return alertas