# Sistema de Analise de Detecção de Quedas - TCC

Um sistema baseado em visão computacional para uma Prova de Conceito para o Trabalho de Conclusão do Curso, do Curso Superior de Tecnologia em Gestão de Dados - CSTGD, do Centro de Educação Aberta e a Distancia - CEAD, da Universidade Federal do Piauí - UFPI

Esse sistema usa o modelo YOLOv11 da Ultralytics para fazer analise de vídeos onde ocorrem situações de risco, como quedas e proximidade.

Ele é composto por um arquivo base que foi a primeira idealização do projeto no qual era usado no Google Colab para execução que posteriormente foi passado para desenvolvimento local para menor gargalo de performance. Onde foi criado os modelos "Sem Velocidade" e o "Com Velocidade" que possui uma trava de segurança para servir de prova no artigo escrito.

Para executar o código é preciso baixar a biblioteca da Ultralytics:

```
pip install ultralytics openCV-python numpy
```

E para execução em lote com vários vídeos use o código abaixo:

```
$modelos = @(
    "modelos\yolo11n.pt",
    "modelos\yolo11s.pt",
    "modelos\yolo11m.pt",
    "modelos\yolo11l.pt",
    "modelos\yolo11x.pt"
)

$videos = Get-ChildItem "videos\*.mp4"

foreach ($video in $videos) {
    Write-Host "Iniciando análise do vídeo: $($video.FullName)" -ForegroundColor Green

    foreach ($modelo in $modelos) {
        python main.py --SOURCE $video.FullName --MODEL $modelo
    }
}
```

Ps. Lembre-se colocar os vídeos na pasta de vídeos, já as pastas de relatórios e modelos o próprio código cria.

## Executando a Interface Web (Streamlit)

O projeto conta com uma interface gráfica completa e interativa para demonstração e análise científica dos resultados do TCC:

```powershell
# Ativar o ambiente virtual e iniciar a aplicação
.\.venv\Scripts\streamlit run app.py
```

A interface disponibiliza:
1. **Visão Geral & Artigo SBC**: Fundamentação teórica, resumo da pesquisa e justificativas éticas.
2. **Estúdio de Inferência**: Execução de vídeos/webcam com comparador lado a lado (**Com Velocidade** vs. **Sem Velocidade**) e ajuste em tempo real de limiares.
3. **Dashboard Científico**: Visualização interativa das métricas empíricas consolidadas (Tabelas 1 a 4 do artigo), matrizes de confusão e gráficos comparativos no Plotly.
4. **Auditoria & Logs**: Registro detalhado das ocorrências com exportação para CSV, Excel e PDF técnico.
5. **Laboratório Experimental**: Simulação da velocidade normalizada para mitigação do Eixo Z e conceitos de Pose Estimation.
