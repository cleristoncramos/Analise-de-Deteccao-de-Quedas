import math

def analisar_postura(largura, altura):
    if largura <= 0:
        return "Indefinido"
    
    ratio = altura / largura

    if ratio < 0.75:
        return "Deitado"
    elif 0.75 <= ratio <= 1.2:
        return "Sentado/Agachado"
    else:
        return "Em pe"
    
def analisar_interacoes(centros, distancia_minima=60):
    alertas = []
    for i in range(len(centros)):
        for j in range(i + 1, len(centros)):
            cx_a, cy_a = centros[i]
            cx_b, cy_b = centros[j]
            distancia = math.sqrt((cx_a - cx_b)**2 + (cy_a - cy_b)**2)
            if distancia < distancia_minima:
                alertas.append((i, j, (cx_a, cy_a), (cx_b, cy_b), distancia))
    return alertas