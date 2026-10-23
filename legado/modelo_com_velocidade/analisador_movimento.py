def classificar_acao(velocidade, limite_andando=1.0, limite_correndo=30.0):
    """
    Transforma a velocidade bruta (pixels/frame) em uma string de ação.
    """
    if velocidade < limite_andando:
        return "Mexendo-se Pouco"
    elif velocidade < limite_correndo:
        return "Andando"
    else:
        return "Correndo"

def confirmar_queda(postura_inferida, velocidade_atual, limite_queda=15.0):
    """
    Uma queda verdadeira é caracterizada pela postura 'Deitado' combinada 
    com um movimento mais brusco (velocidade alta) no momento da transição.
    Se a velocidade for muito baixa, a pessoa apenas deitou de propósito.
    """
    if postura_inferida == "Deitado":
        if velocidade_atual >= limite_queda:
            return True
        else:
            return False
            
    return False