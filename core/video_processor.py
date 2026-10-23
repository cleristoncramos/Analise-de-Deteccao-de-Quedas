import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from .heuristics import analisar_postura, classificar_acao, confirmar_queda

def redimensionar_frame(frame: np.ndarray, largura_alvo: int = 1080) -> np.ndarray:
    """
    Padroniza a largura do frame para neutralizar disparidades de escala espacial (conforme metodologia do TCC).
    """
    altura_original, largura_original = frame.shape[:2]
    if largura_original <= 0 or largura_alvo <= 0 or largura_original == largura_alvo:
        return frame
        
    proporcao = largura_alvo / largura_original
    altura_calculada = int(altura_original * proporcao)
    return cv2.resize(frame, (largura_alvo, altura_calculada))

def _rotulo_id(obj_id) -> str:
    return "?" if obj_id is None else str(obj_id)

def desenhar_anotacoes(
    frame: np.ndarray,
    dados_pessoas: List[Dict[str, Any]],
    alertas_interacoes: List[Dict[str, Any]],
    modo_velocidade: bool = True,
    limiar_queda: float = 15.0,
    limiar_ar: float = 0.75,
    desenhar_trajetoria: bool = True
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Desenha caixas delimitadoras, alertas de risco, informações posturais e linhas de proximidade.
    Retorna o frame anotado e a lista de eventos detectados neste frame.
    """
    frame_desenho = frame.copy()
    eventos_detectados = []
    posturas = [
        analisar_postura(p["largura"], p["altura"], limiar_deitado=limiar_ar)[0] for p in dados_pessoas
    ]

    # 1. Desenhar conexões de proximidade/aglomeração
    for interacao in alertas_interacoes:
        p1 = interacao["ponto_a"]
        p2 = interacao["ponto_b"]
        pm = interacao["ponto_medio"]
        dist = interacao["distancia"]

        # Linha pontilhada/vermelha conectando os dois centroides
        cv2.line(frame_desenho, p1, p2, (0, 0, 255), 2)
        cv2.circle(frame_desenho, pm, 6, (0, 0, 255), -1)
        cv2.putText(
            frame_desenho,
            f"{dist:.1f}px",
            (pm[0] + 8, pm[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 165, 255),
            1,
            cv2.LINE_AA
        )
        pessoa_a = dados_pessoas[interacao["idx_a"]]
        pessoa_b = dados_pessoas[interacao["idx_b"]]
        if modo_velocidade:
            velocidade_txt = f"{pessoa_a['velocidade']:.2f} e {pessoa_b['velocidade']:.2f} px/f"
        else:
            velocidade_txt = "Sem Velocidade"
        eventos_detectados.append({
            "tipo": "PROXIMIDADE",
            "ids": (pessoa_a["id"], pessoa_b["id"]),
            "detalhes": f"Distancia: {dist:.1f}px entre IDs {_rotulo_id(pessoa_a['id'])} e {_rotulo_id(pessoa_b['id'])}",
            "postura": f"{posturas[interacao['idx_a']]} e {posturas[interacao['idx_b']]}",
            "velocidade": velocidade_txt
        })

    # 2. Desenhar cada pessoa detectada
    for pessoa in dados_pessoas:
        x1, y1, x2, y2 = map(int, pessoa["box"])
        largura = pessoa["largura"]
        altura = pessoa["altura"]
        velocidade = pessoa["velocidade"]
        obj_id = _rotulo_id(pessoa["id"])
        trajetoria = pessoa.get("trajetoria", [])

        postura, ratio = analisar_postura(largura, altura, limiar_deitado=limiar_ar)
        acao = classificar_acao(velocidade) if modo_velocidade else "N/A"
        
        queda_confirmada = confirmar_queda(
            postura=postura,
            velocidade=velocidade,
            limiar_queda=limiar_queda,
            usar_velocidade=modo_velocidade
        )

        # Cor da caixa: Vermelho para Queda, Laranja para Deitado/Ação controlada sem velocidade, Azul para Normal
        if queda_confirmada:
            cor = (0, 0, 255)  # Vermelho (Alerta Crítico)
            label_topo = f"ALERTA: QUEDA | ID {obj_id}"
            eventos_detectados.append({
                "tipo": "QUEDA",
                "id": pessoa["id"],
                "detalhes": f"ID {obj_id} (AR: {ratio:.2f})",
                "postura": postura,
                "velocidade": f"{velocidade:.2f} px/f" if modo_velocidade else "Sem Velocidade"
            })
        elif postura == "Deitado" and modo_velocidade:
            cor = (0, 140, 255)  # Laranja (Ação voluntária filtrada com sucesso)
            label_topo = f"ID {obj_id} | Deitado ({acao})"
        else:
            cor = (255, 120, 0)  # Azul claro (Normal)
            if modo_velocidade:
                label_topo = f"ID {obj_id} | {postura} ({acao})"
            else:
                label_topo = f"ID {obj_id} | {postura}"

        # Bounding box
        cv2.rectangle(frame_desenho, (x1, y1), (x2, y2), cor, 2)

        # Rastro de trajetória
        if desenhar_trajetoria and len(trajetoria) > 1:
            for pt_idx in range(1, len(trajetoria)):
                cv2.line(frame_desenho, trajetoria[pt_idx - 1], trajetoria[pt_idx], cor, 1)

        # Tag de identificação superior com fundo preenchido para alta legibilidade
        tamanho_texto, _ = cv2.getTextSize(label_topo, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        largura_txt, altura_txt = tamanho_texto
        y_label = max(y1, altura_txt + 10)
        cv2.rectangle(
            frame_desenho,
            (x1, y_label - altura_txt - 8),
            (x1 + largura_txt + 8, y_label + 2),
            cor,
            -1
        )
        cv2.putText(
            frame_desenho,
            label_topo,
            (x1 + 4, y_label - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Rodapé da caixa com telemetria (Velocidade e Razão de Aspecto)
        if modo_velocidade:
            label_rodape = f"v: {velocidade:.1f} px/f | AR: {ratio:.2f}"
            cv2.putText(
                frame_desenho,
                label_rodape,
                (x1, min(frame_desenho.shape[0] - 5, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                cor,
                1,
                cv2.LINE_AA
            )

    return frame_desenho, eventos_detectados
