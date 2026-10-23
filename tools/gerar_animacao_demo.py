"""Gera assets/demo_heuristica.gif, a animação curta exibida na página inicial.

A cena ilustra a heurística bimodal: na primeira parte a pessoa cai (postura
deitada com velocidade alta, alerta disparado); na segunda ela se deita devagar
(postura deitada com velocidade baixa, nenhum alerta).

Uso: python tools/gerar_animacao_demo.py
"""
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

LARGURA, ALTURA = 720, 405
CHAO = 330
COMPRIMENTO = 150  # dos pés à cabeça, em pixels
LIMIAR_AR = 0.75
LIMIAR_VEL = 15.0

PAPEL = (245, 243, 238)
PAPEL_ESCURO = (235, 231, 222)
TINTA = (29, 29, 27)
TINTA_SUAVE = (110, 108, 102)
MARINHO = (31, 58, 95)
VINHO = (123, 45, 38)
VERDE = (79, 109, 90)

SAIDA = Path(__file__).resolve().parent.parent / "assets" / "demo_heuristica.gif"


def _fonte(tamanho, serifa=False):
    nomes = ["georgia.ttf", "DejaVuSerif.ttf"] if serifa else ["arial.ttf", "DejaVuSans.ttf"]
    for nome in nomes:
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default(tamanho)


FONTE = _fonte(15)
FONTE_PEQUENA = _fonte(13)
FONTE_TITULO = _fonte(18, serifa=True)


def _suavizar(t):
    return t * t * (3 - 2 * t)


def _corpo(x_pes, angulo, fase_passo):
    """Pontos do boneco girando em torno dos pés (0 = em pé, 90 = deitado)."""
    rad = math.radians(angulo)
    eixo = (math.sin(rad), -math.cos(rad))  # direção pés -> cabeça
    normal = (math.cos(rad), math.sin(rad))

    def ponto(ao_longo, lateral=0.0):
        return (
            x_pes + eixo[0] * ao_longo + normal[0] * lateral,
            CHAO + eixo[1] * ao_longo + normal[1] * lateral,
        )

    balanco = math.sin(fase_passo) * 18 * (1 - angulo / 90)
    quadril = ponto(70)
    pescoco = ponto(122)
    return {
        "cabeca": ponto(138),
        "pescoco": pescoco,
        "quadril": quadril,
        "pe_e": ponto(0, -balanco),
        "pe_d": ponto(0, balanco),
        "mao_e": ponto(75, -balanco - 10),
        "mao_d": ponto(75, balanco + 10),
    }


def _caixa(pontos):
    xs = [p[0] for p in pontos.values()]
    ys = [p[1] for p in pontos.values()]
    margem = 14
    return min(xs) - margem, min(ys) - margem, max(xs) + margem, max(ys) + 4


def _cena():
    """Lista de (x_pes, angulo, fase_passo, parte) para cada frame."""
    quadros = []
    # Parte 1: caminha e cai bruscamente
    for i in range(22):
        quadros.append((150 + i * 3.2, 0, i * 0.55, 1))
    x = 150 + 21 * 3.2
    for i in range(1, 7):
        quadros.append((x, 90 * _suavizar(i / 6), 0, 1))
    for _ in range(18):
        quadros.append((x, 90, 0, 1))
    # Parte 2: caminha e se deita devagar
    for i in range(18):
        quadros.append((150 + i * 3.2, 0, i * 0.55, 2))
    x = 150 + 17 * 3.2
    for i in range(1, 37):
        quadros.append((x, 90 * _suavizar(i / 36), 0, 2))
    for _ in range(16):
        quadros.append((x, 90, 0, 2))
    return quadros


def _desenhar(x_pes, angulo, fase, parte, centro_anterior, alerta_ativo):
    img = Image.new("RGB", (LARGURA, ALTURA), PAPEL)
    d = ImageDraw.Draw(img)

    # Moldura da "câmera" e chão
    d.rectangle([0, 0, LARGURA - 1, ALTURA - 1], outline=TINTA, width=1)
    d.rectangle([0, 0, LARGURA - 1, 34], fill=PAPEL_ESCURO, outline=TINTA, width=1)
    titulo = "Cena 1 · queda brusca" if parte == 1 else "Cena 2 · deitar-se devagar"
    d.text((12, 8), titulo, font=FONTE_TITULO, fill=TINTA)
    d.text((LARGURA - 190, 10), "câmera 01 · 30 FPS", font=FONTE_PEQUENA, fill=TINTA_SUAVE)
    d.line([20, CHAO + 4, 470, CHAO + 4], fill=TINTA_SUAVE, width=1)

    pts = _corpo(x_pes, angulo, fase)
    x0, y0, x1, y1 = _caixa(pts)
    centro = ((x0 + x1) / 2, (y0 + y1) / 2)
    ar = (y1 - y0) / max(x1 - x0, 1)
    vel = 0.0 if centro_anterior is None else math.dist(centro, centro_anterior)

    deitado = ar < LIMIAR_AR
    queda = deitado and vel >= LIMIAR_VEL
    alerta_ativo = alerta_ativo or queda
    cor_caixa = VINHO if alerta_ativo else MARINHO

    # Boneco
    for a, b in [("pe_e", "quadril"), ("pe_d", "quadril"), ("quadril", "pescoco"),
                 ("mao_e", "pescoco"), ("mao_d", "pescoco")]:
        d.line([pts[a], pts[b]], fill=TINTA, width=5)
    cx, cy = pts["cabeca"]
    d.ellipse([cx - 13, cy - 13, cx + 13, cy + 13], outline=TINTA, width=4, fill=PAPEL)

    # Bounding box e centroide
    d.rectangle([x0, y0, x1, y1], outline=cor_caixa, width=2)
    d.rectangle([x0, y0 - 20, x0 + 92, y0], fill=cor_caixa)
    d.text((x0 + 5, y0 - 18), "pessoa 0.91", font=FONTE_PEQUENA, fill=PAPEL)
    d.ellipse([centro[0] - 4, centro[1] - 4, centro[0] + 4, centro[1] + 4], fill=cor_caixa)

    # Painel de leitura da heurística
    px = 490
    d.rectangle([px, 50, LARGURA - 14, ALTURA - 16], fill=(251, 250, 247), outline=TINTA, width=1)
    d.text((px + 12, 60), "Leitura da heurística", font=FONTE_TITULO, fill=TINTA)
    d.line([px + 12, 86, LARGURA - 26, 86], fill=TINTA_SUAVE, width=1)

    postura = "Deitado" if deitado else ("Sentado" if ar <= 1.2 else "Em pé")
    linhas = [
        ("Razão de aspecto (h/w)", f"{ar:4.2f}", VINHO if deitado else TINTA),
        ("Postura", postura, VINHO if deitado else TINTA),
        ("Velocidade (px/f)", f"{vel:4.1f}", VINHO if vel >= LIMIAR_VEL else TINTA),
    ]
    y = 98
    for rotulo, valor, cor in linhas:
        d.text((px + 12, y), rotulo, font=FONTE_PEQUENA, fill=TINTA_SUAVE)
        d.text((px + 12, y + 17), valor, font=FONTE, fill=cor)
        y += 46

    d.text((px + 12, y), f"Regra: AR < {LIMIAR_AR} e v ≥ {LIMIAR_VEL:.0f}", font=FONTE_PEQUENA, fill=TINTA_SUAVE)
    y += 28
    if alerta_ativo:
        d.rectangle([px + 12, y, LARGURA - 26, y + 44], fill=VINHO)
        d.text((px + 22, y + 13), "QUEDA DETECTADA", font=FONTE, fill=PAPEL)
    elif deitado:
        d.rectangle([px + 12, y, LARGURA - 26, y + 44], outline=VERDE, width=2)
        d.text((px + 22, y + 4), "Deitado sem impacto", font=FONTE, fill=VERDE)
        d.text((px + 22, y + 23), "nenhum alerta", font=FONTE_PEQUENA, fill=VERDE)
    else:
        d.rectangle([px + 12, y, LARGURA - 26, y + 44], outline=MARINHO, width=1)
        d.text((px + 22, y + 13), "Monitorando", font=FONTE, fill=MARINHO)

    return img, centro, alerta_ativo


def gerar(saida=SAIDA):
    quadros = []
    centro_anterior = None
    alerta = False
    parte_anterior = None
    for x_pes, angulo, fase, parte in _cena():
        if parte != parte_anterior:
            centro_anterior, alerta, parte_anterior = None, False, parte
        img, centro_anterior, alerta = _desenhar(x_pes, angulo, fase, parte, centro_anterior, alerta)
        quadros.append(img.quantize(colors=32))
    saida.parent.mkdir(parents=True, exist_ok=True)
    quadros[0].save(saida, save_all=True, append_images=quadros[1:], duration=90, loop=0, optimize=True)
    return saida


if __name__ == "__main__":
    caminho = gerar()
    print(f"Animação salva em {caminho} ({caminho.stat().st_size / 1024:.0f} KB)")
