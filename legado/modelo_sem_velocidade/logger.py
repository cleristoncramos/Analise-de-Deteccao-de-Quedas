import os
import csv
from datetime import datetime

def criar_pastas_video(nome_video, nome_modelo=''):
    nome_vid_base = os.path.splitext(nome_video)[0]
    nome_mod_base = os.path.splitext(nome_modelo)[0] if nome_modelo else 'modelo'

    
    pasta_relatorio = os.path.join('relatorios', nome_vid_base, nome_mod_base)
    os.makedirs(pasta_relatorio, exist_ok=True)
    
    pasta_video = os.path.join('videos', nome_vid_base, nome_mod_base)
    os.makedirs(pasta_video, exist_ok=True)
    
    return pasta_relatorio, pasta_video

def gerar_nome_unico(caminho_arquivo):
    
    if not os.path.exists(caminho_arquivo):
        return caminho_arquivo

    pasta = os.path.dirname(caminho_arquivo)
    nome_completo = os.path.basename(caminho_arquivo)

    nome, extensao = os.path.splitext(nome_completo)

    contador = 1

    while True:
        novo_nome = f"{nome}_{contador}{extensao}"
        novo_caminho = os.path.join(pasta, novo_nome)

        if not os.path.exists(novo_caminho):
            return novo_caminho

        contador += 1

def iniciar_log(nome_video='video_padrao', nome_modelo=''):

    pasta_relatorio, _ = criar_pastas_video(nome_video, nome_modelo)
    
    nome_vid = os.path.splitext(nome_video)[0]
    nome_mod = os.path.splitext(nome_modelo)[0] if nome_modelo else 'modelo'
    
    nome_arquivo_csv = f"relatorio_{nome_vid}_{nome_mod}.csv"

    nome_arquivo = os.path.join(pasta_relatorio, nome_arquivo_csv)

    nome_arquivo = gerar_nome_unico(nome_arquivo)
    
    arquivo = open(nome_arquivo, mode='w', newline='', encoding='utf-8')

    escritor = csv.writer(arquivo)

    escritor.writerow([
        'Frame',
        'Data_Hora', 
        'Tipo_Evento', 
        'Detalhes', 
        'Postura_Inferida'
    ])

    return arquivo, escritor

def registrar_evento(escritor, frame, tipo, detalhes, postura="N/A"):
    agora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    escritor.writerow([frame, agora, tipo, detalhes, postura])

def fechar_log(arquivo):
    arquivo.close()