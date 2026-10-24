# Sistema de Analise de Detecção de Quedas - TCC

Um sistema baseado em visão computacional para uma Prova de Conceito para o Trabalho de Conclusão do Curso, do Curso Superior de Tecnologia em Gestão de Dados - CSTGD, do Centro de Educação Aberta e a Distancia - CEAD, da Universidade Federal do Piauí - UFPI

Esse sistema usa o modelo YOLOv11 da Ultralytics para fazer analise de vídeos onde ocorrem situações de risco, como quedas e proximidade.

A primeira versão do projeto (rodada no Google Colab) e os modelos "Sem Velocidade" e "Com Velocidade" usados nos experimentos do artigo estão em `legado/`. O pipeline atual fica em `core/` e é compartilhado pela linha de comando (`cli.py`) e pela interface web (`app.py`).

## Instalação

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Os pesos (`yolo11n.pt` … `yolo11x.pt`) são procurados em `modelos/` e na raiz do projeto; se não existirem, a Ultralytics baixa os oficiais automaticamente.

## Linha de comando

```bash
# Proposta: postura + velocidade
python cli.py --SOURCE videos/queda_01.mp4 --MODEL yolo11s.pt --MODO com
# Linha de base: apenas postura
python cli.py --SOURCE videos/queda_01.mp4 --MODEL yolo11s.pt --MODO sem
# Webcam
python cli.py --SOURCE 0 --MODEL yolo11n.pt
```

Os relatórios CSV são gravados em `relatorios/<video>/<modelo>/`. Opções úteis (`python cli.py --help` lista todas):

- `--SEM-JANELA`: processa sem abrir a janela de vídeo (mais rápido em lote).
- `--POR-FRAME`: registra todo frame que satisfaz a heurística, como nos scripts originais. Por padrão, a mesma pessoa (ou dupla, na proximidade) só gera um novo evento depois de `--CARENCIA` segundos (padrão 2 s).
- `--SEM-CORRECAO-FPS`: usa a velocidade em pixels por frame sem converter para a referência de 30 FPS.

Execução em lote (PowerShell):

```powershell
$modelos = @("yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt")
foreach ($video in Get-ChildItem "videos\*.mp4") {
    foreach ($modelo in $modelos) {
        foreach ($modo in @("com", "sem")) {
            python cli.py --SOURCE $video.FullName --MODEL $modelo --MODO $modo --SEM-JANELA
        }
    }
}
```

Equivalente em bash:

```bash
for video in videos/*.mp4; do
  for modelo in yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt; do
    for modo in com sem; do
      python cli.py --SOURCE "$video" --MODEL "$modelo" --MODO "$modo" --SEM-JANELA
    done
  done
done
```

Os scripts originais continuam em `legado/` e devem ser executados de dentro da própria pasta (ex.: `cd legado/modelo_com_velocidade && python main.py --SOURCE ../../videos/x.mp4 --MODEL ../../modelos/yolo11s.pt`).

## Executando a Interface Web (Streamlit)

O projeto conta com uma interface gráfica completa e interativa para demonstração e análise científica dos resultados do TCC:

```bash
streamlit run app.py
```

Para publicar no Streamlit Community Cloud, o arquivo `packages.txt` instala as bibliotecas de sistema `libgl1` e `libglib2.0-0`, exigidas pelo OpenCV, que é dependência da Ultralytics.

A interface disponibiliza:
1. **Visão Geral & Artigo SBC**: Fundamentação teórica, resumo da pesquisa e justificativas éticas.
2. **Estúdio de Inferência**: Execução de vídeos/webcam com comparador lado a lado (**Com Velocidade** vs. **Sem Velocidade**) e ajuste em tempo real de limiares.
3. **Dashboard Científico**: Visualização interativa das métricas empíricas consolidadas (Tabelas 1 a 4 do artigo), matrizes de confusão e gráficos comparativos no Plotly.
4. **Auditoria & Logs**: Registro detalhado das ocorrências com exportação para CSV, Excel e PDF técnico.
5. **Laboratório Experimental**: Simulação da velocidade normalizada para mitigação do Eixo Z e conceitos de Pose Estimation.

### Câmera do navegador no app publicado (TURN)

A opção **Câmera do Navegador** do Estúdio de Inferência usa WebRTC. Localmente basta o STUN público, mas no Streamlit Community Cloud o servidor fica atrás de NAT e a conexão de vídeo precisa de um servidor TURN; sem ele, a câmera fica em "Connection is taking longer than expected" com a imagem preta.

1. Crie uma conta gratuita em [metered.ca](https://www.metered.ca/), abra **TURN Server** e anote o nome do app (o subdomínio de `<app>.metered.live`) e a **API key**.
2. No Streamlit Cloud, abra **Manage app > Settings > Secrets** e cole:

   ```toml
   [turn]
   metered_app = "nome-do-app"
   metered_api_key = "sua-api-key"
   ```

3. Salve e reinicie o app. A legenda abaixo da opção passa a indicar que a conexão usa TURN.

Também é possível usar Twilio ou uma lista fixa de servidores; o modelo com todos os campos está em `.streamlit/secrets.toml.example`. Para testar localmente, copie esse arquivo para `.streamlit/secrets.toml`, que não é versionado.

## Observações sobre os dados

Dois pontos das tabelas do artigo (`data/tcc_benchmark_data.json`) ainda precisam ser conferidos na contagem original. Eles ficam registrados no campo `metadados.pendencias` e não são exibidos na interface:

- **YOLOv11x (Com Velocidade):** soma 80 eventos. Na Tabela 1 tem 17 eventos simulados (10 negativos), enquanto os demais modelos têm 16 (9 negativos).
- **Sem Velocidade (YOLOv11m\*):** soma 91 eventos (53 negativos, contra 41 nos demais). Os falsos alarmes parecem ter sido contados como eventos novos; para comparar a acurácia entre cenários, os dois devem usar o mesmo conjunto de eventos reais.

## Testes

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

Os testes cobrem as heurísticas, o rastreador, o agrupamento de eventos e a consistência das métricas em `data/tcc_benchmark_data.json`. O GitHub Actions roda os dois comandos a cada push e pull request.
