"""Rótulos e chaves de estado compartilhados entre as páginas (menu, fontes de vídeo e modos de análise)."""

MODULO_ESTUDIO = "Estúdio de Inferência"
MODULO_AUDITORIA = "Auditoria e Logs"
MODULO_VISAO_GERAL = "Sobre a pesquisa"

MODO_COM = "Bimodal (Com Velocidade - Proposta)"
MODO_SEM = "Estático (Sem Velocidade - Linha de Base)"
MODO_AB = "Comparativo Lado a Lado (A/B: Sem vs Com Velocidade)"

FONTE_AMOSTRA = "Amostra Pré-Carregada"
FONTE_UPLOAD = "Upload de Arquivo de Vídeo"
FONTE_LINK = "Link de Vídeo ou Transmissão"
FONTE_NAVEGADOR = "Câmera do Navegador"
FONTE_SERVIDOR = "Webcam do Servidor (uso local)"

# Chaves de st.session_state dos widgets que a página inicial pré-seleciona
CHAVE_MODULO = "modulo_sistema"
CHAVE_FONTE = "fonte_entrada"
CHAVE_MODO = "modo_analise"
CHAVE_AMOSTRA_PADRAO = "amostra_padrao_id"

AMOSTRA_EXEMPLO = "fall-01"  # queda detectada nos dois modos: mostra a ferramenta funcionando
