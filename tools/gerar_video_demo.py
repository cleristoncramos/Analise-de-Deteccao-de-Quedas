"""Gera assets/video/apresentacao.mp4, o vídeo curto exibido na página inicial.

Simula a gravação de uma câmera de segurança em sala de aula, com as caixas e
leituras do próprio sistema:
- cena 1: um aluno tropeça e cai (postura deitada com velocidade alta, alerta);
- cena 2: outro aluno se deita devagar (postura deitada com velocidade baixa, sem alerta).

As leituras de razão de aspecto (AR) e velocidade (v) são calculadas sobre as
caixas desenhadas, com a mesma regra do sistema: AR < 0,75 e v >= 15 px/f.

Uso: python tools/gerar_video_demo.py

Com o pacote imageio-ffmpeg instalado (pip install imageio-ffmpeg), o vídeo é
comprimido em H.264 com o ffmpeg dele (arquivo de poucos MB); sem ele, usa o
codificador H.264 do próprio OpenCV, que gera um arquivo bem maior.
"""
import math
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

W, H = 1280, 720
FPS = 24
LIMIAR_AR = 0.75
LIMIAR_VEL = 15.0
FUNDO_Y, FRENTE_Y = 330, 720  # linha do rodapé da parede do fundo e borda inferior do chão
SAIDA = Path(__file__).resolve().parent.parent / "assets" / "video" / "apresentacao.mp4"

# Cores em BGR
VINHO = (38, 45, 123)
MARINHO = (95, 58, 31)
AZUL_CAIXA = (200, 140, 60)
BRANCO = (240, 240, 240)
PRETO = (20, 20, 20)
PELE = (125, 160, 205)
CABELO = (30, 40, 55)
SAPATO = (25, 25, 30)

rng = np.random.default_rng(7)


def _poligono(img, pts, cor):
    cv2.fillPoly(img, [np.array(pts, np.int32)], cor, lineType=cv2.LINE_AA)


def _gradiente_vertical(h, w, cima, baixo):
    t = np.linspace(0, 1, h)[:, None, None]
    return (np.array(cima) * (1 - t) + np.array(baixo) * t).repeat(w, axis=1)


def _escala(y_pes):
    """Perspectiva: quanto mais perto da câmera (y maior), maior a pessoa."""
    return 0.35 + 0.65 * (y_pes - FUNDO_Y) / (FRENTE_Y - FUNDO_Y)


def _carteira(img, x, y, s):
    """Carteira escolar simples em perspectiva, com o pé apoiado em (x, y)."""
    lw, lh, prof, alt = 120 * s, 10 * s, 45 * s, 70 * s
    tampo = [(x, y - alt), (x + lw, y - alt), (x + lw - prof * 0.4, y - alt - prof * 0.5),
             (x - prof * 0.4, y - alt - prof * 0.5)]
    _poligono(img, [(x + 6 * s, y - alt), (x + 10 * s, y - alt), (x + 10 * s, y), (x + 6 * s, y)], (70, 70, 75))
    _poligono(img, [(x + lw - 10 * s, y - alt), (x + lw - 6 * s, y - alt), (x + lw - 6 * s, y),
                    (x + lw - 10 * s, y)], (70, 70, 75))
    _poligono(img, tampo, (95, 135, 170))
    _poligono(img, [(x, y - alt), (x + lw, y - alt), (x + lw, y - alt + lh), (x, y - alt + lh)], (60, 95, 125))
    cv2.ellipse(img, (int(x + lw / 2), int(y + 2)), (int(lw * 0.55), int(8 * s)), 0, 0, 360, (150, 150, 145), -1,
                cv2.LINE_AA)


def _cenario():
    img = np.zeros((H, W, 3), np.float64)
    # Teto, paredes e chão
    img[:] = _gradiente_vertical(H, W, (200, 205, 210), (185, 190, 195))
    _poligono(img, [(0, 0), (W, 0), (1050, 60), (230, 60)], (215, 218, 220))
    parede_fundo = _gradiente_vertical(FUNDO_Y - 60, 820, (190, 205, 215), (165, 180, 192))
    img[60:FUNDO_Y, 230:1050] = parede_fundo
    _poligono(img, [(0, 0), (230, 60), (230, FUNDO_Y), (0, FRENTE_Y - 120)], (160, 175, 188))
    _poligono(img, [(W, 0), (1050, 60), (1050, FUNDO_Y), (W, FRENTE_Y - 120)], (175, 188, 200))

    chao = np.zeros_like(img)
    chao[:] = _gradiente_vertical(H, W, (150, 160, 165), (185, 192, 196))
    mascara = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mascara, [np.array([(230, FUNDO_Y), (1050, FUNDO_Y), (W, FRENTE_Y - 120), (W, H), (0, H),
                                     (0, FRENTE_Y - 120)], np.int32)], 1)
    img[mascara == 1] = chao[mascara == 1]

    # Rejunte do piso em perspectiva
    ponto_fuga = (640, -260)
    for i in range(-14, 15):
        x_fundo = 640 + i * 55
        dx = x_fundo - ponto_fuga[0]
        dy = FUNDO_Y - ponto_fuga[1]
        x_frente = ponto_fuga[0] + dx * (H - ponto_fuga[1]) / dy
        cv2.line(img, (int(x_fundo), FUNDO_Y), (int(x_frente), H), (140, 148, 152), 1, cv2.LINE_AA)
    for k in range(1, 12):
        y = FUNDO_Y + (H - FUNDO_Y) * (k / 12) ** 1.6
        cv2.line(img, (0, int(y)), (W, int(y)), (140, 148, 152), 1, cv2.LINE_AA)
    cv2.line(img, (230, FUNDO_Y), (1050, FUNDO_Y), (110, 115, 120), 2, cv2.LINE_AA)
    cv2.line(img, (230, FUNDO_Y), (0, FRENTE_Y - 120), (110, 115, 120), 2, cv2.LINE_AA)
    cv2.line(img, (1050, FUNDO_Y), (W, FRENTE_Y - 120), (110, 115, 120), 2, cv2.LINE_AA)

    # Quadro, janela e porta
    cv2.rectangle(img, (420, 105), (860, 240), (120, 120, 125), -1)
    cv2.rectangle(img, (428, 112), (852, 233), (55, 80, 50), -1)
    for i, (x0, y0, x1) in enumerate([(450, 135, 640), (450, 160, 700), (450, 185, 600), (700, 140, 820)]):
        cv2.line(img, (x0, y0), (x1, y0 + (i % 2) * 3), (190, 200, 195), 2, cv2.LINE_AA)
    cv2.rectangle(img, (420, 240), (860, 246), (140, 140, 140), -1)
    _poligono(img, [(1095, 70), (1230, 40), (1230, 330), (1095, 300)], (235, 225, 205))
    cv2.line(img, (1162, 55), (1162, 315), (150, 150, 150), 3, cv2.LINE_AA)
    cv2.line(img, (1095, 185), (1230, 185), (150, 150, 150), 3, cv2.LINE_AA)
    _poligono(img, [(120, 170), (200, 130), (200, 335), (120, 420)], (70, 95, 125))
    cv2.circle(img, (185, 285), 4, (60, 160, 200), -1, cv2.LINE_AA)

    # Luz da janela no chão e carteiras ao fundo
    luz = np.zeros_like(img)
    _poligono(luz, [(880, 360), (1050, 340), (1240, 520), (980, 560)], (35, 35, 30))
    img += cv2.GaussianBlur(luz, (0, 0), 25)
    for y in (385, 455):
        s = _escala(y)
        for x in (270, 420):
            _carteira(img, x - (y - 385) * 0.6, y, s)
        for x in (760, 900):
            _carteira(img, x + (y - 385) * 0.6, y, s)

    # Vinheta da lente
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= (1 - 0.28 * np.clip(dist - 0.35, 0, 1))[..., None]
    return cv2.GaussianBlur(np.clip(img, 0, 255), (0, 0), 0.8)


# ----------------------------------------------------------------- pessoa


class Pessoa:
    """Boneco articulado visto de perfil, andando para a direita."""

    def __init__(self, camisa, calca, altura=330):
        self.camisa, self.calca, self.altura = camisa, calca, altura

    def esqueleto(self, fase, agacho=0.0, bracos_frente=0.0):
        """Pontos locais (pés no chão em y=0, para cima é y negativo), em unidades da altura 330."""
        coxa, canela, tronco, braco, antebraco = 85, 85, 105, 60, 55
        amp = 28 * (1 - agacho)
        pernas = []
        for sinal in (1, -1):
            a_coxa = math.radians(amp * math.sin(fase) * sinal + 75 * agacho)
            dobra = math.radians(max(0.0, 40 * math.sin(fase + sinal * 1.2)) * (1 - agacho) + 110 * agacho)
            joelho = (coxa * math.sin(a_coxa), coxa * math.cos(a_coxa))
            a_canela = a_coxa - dobra
            pe = (joelho[0] + canela * math.sin(a_canela), joelho[1] + canela * math.cos(a_canela))
            pernas.append((joelho, pe))
        piso = max(p[1][1] for p in pernas)
        quadril = (0.0, -piso)
        inclinacao = math.radians(8 + 25 * agacho)
        pescoco = (quadril[0] + tronco * math.sin(inclinacao), quadril[1] - tronco * math.cos(inclinacao))
        cabeca = (pescoco[0] + 14 * math.sin(inclinacao) + 4, pescoco[1] - 38)
        bracos = []
        for sinal in (1, -1):
            a = math.radians(-30 * math.sin(fase) * sinal * (1 - bracos_frente) + 150 * bracos_frente
                             + 20 * agacho)
            cotovelo = (pescoco[0] + braco * math.sin(a), pescoco[1] + 8 + braco * math.cos(a))
            a2 = a + math.radians(25 + 20 * bracos_frente)
            mao = (cotovelo[0] + antebraco * math.sin(a2), cotovelo[1] + antebraco * math.cos(a2))
            bracos.append((cotovelo, mao))
        pts = {"quadril": quadril, "pescoco": pescoco, "cabeca": cabeca}
        for nome, (joelho, pe) in zip(("d", "e"), pernas):
            pts[f"joelho_{nome}"] = (joelho[0], joelho[1] + quadril[1])
            pts[f"pe_{nome}"] = (pe[0], pe[1] + quadril[1])
        for nome, (cot, mao) in zip(("d", "e"), bracos):
            pts[f"cotovelo_{nome}"], pts[f"mao_{nome}"] = cot, mao
        return pts

    def projetar(self, pts, x, y, s, angulo, pivo_x=0.0):
        """Gira o corpo em torno do ponto de apoio (queda para a frente) e aplica a perspectiva."""
        t = math.radians(angulo)
        k = s * self.altura / 330
        saida = {}
        for nome, (px, py) in pts.items():
            px -= pivo_x
            rx = px * math.cos(t) - py * math.sin(t)
            ry = px * math.sin(t) + py * math.cos(t)
            ry -= 22 * math.sin(t)  # espessura do corpo deitado
            saida[nome] = (x + (rx + pivo_x) * k, y + ry * k)
        return saida, k

    def desenhar(self, img, p, k):
        def seg(a, b, cor, esp):
            cv2.line(img, tuple(int(v) for v in p[a]), tuple(int(v) for v in p[b]), cor, max(2, int(esp * k)),
                     cv2.LINE_AA)

        escuro = tuple(int(c * 0.75) for c in self.calca)
        camisa_escura = tuple(int(c * 0.8) for c in self.camisa)
        pele_escura = tuple(int(c * 0.85) for c in PELE)
        # Membros do lado de trás
        seg("quadril", "joelho_e", escuro, 26)
        seg("joelho_e", "pe_e", escuro, 22)
        cv2.circle(img, tuple(int(v) for v in p["pe_e"]), int(12 * k), SAPATO, -1, cv2.LINE_AA)
        seg("pescoco", "cotovelo_e", camisa_escura, 18)
        seg("cotovelo_e", "mao_e", pele_escura, 14)
        # Tronco e cabeça
        seg("quadril", "pescoco", self.camisa, 46)
        cx, cy = (int(v) for v in p["cabeca"])
        cv2.circle(img, (cx, cy), int(27 * k), PELE, -1, cv2.LINE_AA)
        cv2.ellipse(img, (cx - int(4 * k), cy - int(6 * k)), (int(28 * k), int(22 * k)), 0, 160, 380, CABELO, -1,
                    cv2.LINE_AA)
        # Membros da frente
        seg("quadril", "joelho_d", self.calca, 26)
        seg("joelho_d", "pe_d", self.calca, 22)
        cv2.circle(img, tuple(int(v) for v in p["pe_d"]), int(12 * k), SAPATO, -1, cv2.LINE_AA)
        seg("pescoco", "cotovelo_d", self.camisa, 18)
        seg("cotovelo_d", "mao_d", PELE, 14)

    @staticmethod
    def caixa(p, k):
        xs = [v[0] for v in p.values()]
        ys = [v[1] for v in p.values()]
        m = 30 * k
        return min(xs) - m, min(ys) - m, max(xs) + m * 0.6, max(ys) + m * 0.4


def _sombra(img, x, y, s, largura):
    camada = np.zeros_like(img)
    cv2.ellipse(camada, (int(x), int(y + 4)), (int(largura), int(14 * s)), 0, 0, 360, (60, 60, 60), -1, cv2.LINE_AA)
    img -= cv2.GaussianBlur(camada, (0, 0), 6) * 0.6


# ------------------------------------------------------------- sobreposições


def _texto(img, txt, org, escala=0.55, cor=BRANCO, esp=1, fundo=None):
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(txt, fonte, escala, esp)
    if fundo is not None:
        cv2.rectangle(img, (org[0] - 6, org[1] - th - 7), (org[0] + tw + 6, org[1] + base + 3), fundo, -1)
    cv2.putText(img, txt, org, fonte, escala, cor, esp, cv2.LINE_AA)
    return tw


def _caixa_deteccao(img, caixa, cor, rotulo, rodape):
    x0, y0, x1, y1 = (int(v) for v in caixa)
    cv2.rectangle(img, (x0, y0), (x1, y1), cor, 2, cv2.LINE_AA)
    for (cx, cy), (dx, dy) in [((x0, y0), (1, 1)), ((x1, y0), (-1, 1)), ((x0, y1), (1, -1)), ((x1, y1), (-1, -1))]:
        cv2.line(img, (cx, cy), (cx + 18 * dx, cy), cor, 4, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + 18 * dy), cor, 4, cv2.LINE_AA)
    _texto(img, rotulo, (x0 + 6, y0 - 9), 0.55, BRANCO, 1, cor)
    _texto(img, rodape, (x0 + 6, y1 + 22), 0.5, BRANCO, 1, (40, 40, 40))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    cv2.drawMarker(img, (cx, cy), cor, cv2.MARKER_CROSS, 14, 2, cv2.LINE_AA)


def _hud(img, n, inicio, cena):
    faixa = img.copy()
    cv2.rectangle(faixa, (0, 0), (W, 44), (15, 15, 15), -1)
    cv2.addWeighted(faixa, 0.7, img, 0.3, 0, img)
    _texto(img, "CAM 01  |  SALA 03 - BLOCO B", (18, 29), 0.62)
    hora = inicio + timedelta(seconds=n / FPS)
    _texto(img, hora.strftime("%d/%m/%Y  %H:%M:%S"), (W - 300, 29), 0.62)
    if (n // 12) % 2 == 0:
        cv2.circle(img, (W - 330, 23), 7, (40, 40, 220), -1, cv2.LINE_AA)
    _texto(img, cena, (18, H - 20), 0.55, BRANCO, 1, (30, 30, 30))
    _texto(img, "YOLOv11s  |  postura + velocidade  |  24 FPS", (W - 430, H - 20), 0.5, (220, 220, 220), 1,
           (30, 30, 30))


def _cartao(img, titulo, subtitulo, alfa):
    escuro = np.zeros_like(img)
    cv2.addWeighted(escuro, alfa * 0.75, img, 1 - alfa * 0.75, 0, img)
    if alfa > 0.4:
        fonte = cv2.FONT_HERSHEY_DUPLEX
        (tw, _), _ = cv2.getTextSize(titulo, fonte, 1.3, 2)
        cv2.putText(img, titulo, ((W - tw) // 2, H // 2 - 10), fonte, 1.3, BRANCO, 2, cv2.LINE_AA)
        (tw, _), _ = cv2.getTextSize(subtitulo, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        cv2.putText(img, subtitulo, ((W - tw) // 2, H // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (210, 210, 210), 1,
                    cv2.LINE_AA)


# ------------------------------------------------------------------- roteiro


def _roteiro():
    """Lista de estados por quadro: (cena, x, angulo, fase, agacho, bracos, pivo, cartao)."""
    quadros = []
    # Cena 1: caminha, tropeça e cai de forma brusca (aceleração da gravidade)
    for i in range(22):
        quadros.append(("c1", 120, 0, 0, 0, 0, 0, 1 - i / 22))
    x = 120.0
    for i in range(64):
        x += 4.6
        quadros.append(("c1", x, 0, i * 0.42, 0, 0, 0, 0))
    fase = 64 * 0.42
    for i in range(1, 10):
        t = i / 9
        quadros.append(("c1", x, 88 * t * t, fase * (1 - t) + math.pi * 20 * t, 0, min(1, t * 1.6), 40, 0))
    for ang in (93, 89, 90, 90):
        quadros.append(("c1", x, ang, math.pi * 20, 0.15, 1, 40, 0))
    for _ in range(62):
        quadros.append(("c1", x, 90, math.pi * 20, 0.15, 1, 40, 0))
    for i in range(10):
        quadros.append(("c1", x, 90, math.pi * 20, 0.15, 1, 40, i / 10))
    # Cena 2: caminha, senta e se deita devagar
    for i in range(22):
        quadros.append(("c2", 160, 0, 0, 0, 0, 0, 1 - i / 22))
    x = 160.0
    for i in range(52):
        x += 4.6
        quadros.append(("c2", x, 0, i * 0.42, 0, 0, 0, 0))
    fase = 52 * 0.42
    for i in range(1, 37):
        t = i / 36
        quadros.append(("c2", x, 0, fase * (1 - t) + math.pi * 16 * t, 0.9 * t, 0, 0, 0))
    for i in range(1, 61):
        t = i / 60
        quadros.append(("c2", x, 90 * (t * t * (3 - 2 * t)), math.pi * 16, 0.9 - 0.75 * t, 0.9 * t, 60, 0))
    for _ in range(56):
        quadros.append(("c2", x, 90, math.pi * 16, 0.15, 0.9, 60, 0))
    return quadros


class _Escritor:
    """Grava quadros BGR em MP4 H.264, com ffmpeg (imageio-ffmpeg) quando disponível."""

    def __init__(self, saida, tamanho, crf=27):
        self.tamanho = tamanho
        try:
            import imageio_ffmpeg
        except ImportError:
            self.ffmpeg = None
            self.cv = cv2.VideoWriter(str(saida), cv2.VideoWriter_fourcc(*"avc1"), FPS, tamanho)
            if not self.cv.isOpened():
                raise RuntimeError("Sem imageio-ffmpeg e sem suporte a H.264 (avc1) no OpenCV")
        else:
            self.ffmpeg = imageio_ffmpeg.write_frames(
                str(saida), tamanho, fps=FPS, codec="libx264", pix_fmt_in="bgr24", quality=None,
                macro_block_size=8,
                output_params=["-crf", str(crf), "-preset", "slow", "-movflags", "+faststart"])
            self.ffmpeg.send(None)

    def write(self, quadro):
        quadro = cv2.resize(quadro, self.tamanho, interpolation=cv2.INTER_AREA)
        if self.ffmpeg is not None:
            self.ffmpeg.send(np.ascontiguousarray(quadro))
        else:
            self.cv.write(quadro)

    def release(self):
        if self.ffmpeg is not None:
            self.ffmpeg.close()
        else:
            self.cv.release()


def gerar(saida=SAIDA, tamanho=(960, 540)):
    fundo = _cenario()
    grao = rng.normal(0, 2.5, (H, W, 1))
    aluno1 = Pessoa(camisa=(235, 235, 235), calca=(110, 60, 35))
    aluno2 = Pessoa(camisa=(120, 150, 70), calca=(70, 70, 75))
    colega = Pessoa(camisa=(70, 110, 190), calca=(90, 90, 95))
    y_pes = 575
    s = _escala(y_pes)
    inicio = datetime(2026, 10, 23, 10, 42, 5)

    saida.parent.mkdir(parents=True, exist_ok=True)
    escritor = _Escritor(saida, tamanho)

    centro_ant, cena_ant, alerta, anterior = None, None, False, None
    for n, (cena, x, ang, fase, agacho, bracos, pivo, cartao) in enumerate(_roteiro()):
        if cena != cena_ant:
            centro_ant, alerta, cena_ant = None, False, cena
        img = fundo.copy()

        # Colega ao fundo, passando da direita para a esquerda
        yc = 395
        sc = _escala(yc)
        xc = 1180 - (n % 300) * 3.2
        pc, kc = colega.projetar(colega.esqueleto(n * 0.42), xc, yc, sc, 0)
        pc = {nome: (2 * xc - px, py) for nome, (px, py) in pc.items()}  # espelha: anda para a esquerda
        _sombra(img, xc, yc, sc, 40 * sc)
        colega.desenhar(img, pc, kc)

        aluno = aluno1 if cena == "c1" else aluno2
        pts, k = aluno.projetar(aluno.esqueleto(fase, agacho, bracos), x, y_pes, s, ang, pivo)
        _sombra(img, x + 110 * k * math.sin(math.radians(ang)), y_pes, s, (55 + 120 * ang / 90) * k)
        aluno.desenhar(img, pts, k)

        # Câmera: borrão de movimento, granulado e oscilação de brilho
        if anterior is not None and 10 < ang < 92 and cena == "c1":
            img = img * 0.62 + anterior * 0.38
        anterior = img.copy()
        img = img * (1 + 0.012 * math.sin(n * 0.9)) + grao
        quadro = np.clip(img, 0, 255).astype(np.uint8)

        # Leituras do sistema
        caixa = Pessoa.caixa(pts, k)
        centro = ((caixa[0] + caixa[2]) / 2, (caixa[1] + caixa[3]) / 2)
        ar = (caixa[3] - caixa[1]) / max(caixa[2] - caixa[0], 1)
        v = 0.0 if centro_ant is None else math.dist(centro, centro_ant) * FPS / 30
        centro_ant = centro
        deitado = ar < LIMIAR_AR
        alerta = alerta or (deitado and v >= LIMIAR_VEL)
        if alerta:
            cor, rotulo = VINHO, "ID 3  pessoa 0.92  ALERTA: QUEDA"
        elif deitado:
            cor, rotulo = AZUL_CAIXA, "ID 7  pessoa 0.90  deitado - sem alerta"
        else:
            cor, rotulo = AZUL_CAIXA, f"ID {3 if cena == 'c1' else 7}  pessoa 0.94"
        _caixa_deteccao(quadro, caixa, cor, rotulo, f"AR {ar:4.2f}   v {v:4.1f} px/f")
        cbox = Pessoa.caixa(pc, kc)
        _caixa_deteccao(quadro, cbox, AZUL_CAIXA, "ID 5  pessoa 0.88", "AR 2.6   v 2.6 px/f")

        if alerta and (n // 8) % 2 == 0:
            cv2.rectangle(quadro, (W // 2 - 260, 62), (W // 2 + 260, 112), VINHO, -1)
            _texto(quadro, "ALERTA: QUEDA DETECTADA - ID 3", (W // 2 - 225, 96), 0.85, BRANCO, 2)
        elif deitado and not alerta:
            cv2.rectangle(quadro, (W // 2 - 300, 62), (W // 2 + 300, 112), (60, 80, 60), -1)
            _texto(quadro, "DEITADO COM VELOCIDADE BAIXA - SEM ALERTA", (W // 2 - 275, 96), 0.75, BRANCO, 2)

        nome_cena = "CENA 1 - QUEDA BRUSCA" if cena == "c1" else "CENA 2 - DEITAR-SE DEVAGAR"
        _hud(quadro, n, inicio, nome_cena)
        if cartao > 0:
            sub = ("AR < 0,75 e v >= 15 px/f: alerta" if cena == "c1"
                   else "AR < 0,75 com v baixa: nenhum alerta").replace(",", ".")
            _cartao(quadro, nome_cena.title().replace(" - ", ": "), sub, cartao)
        escritor.write(quadro)
    escritor.release()
    return saida


if __name__ == "__main__":
    caminho = gerar()
    print(f"Vídeo salvo em {caminho} ({caminho.stat().st_size / 1024:.0f} KB)")
