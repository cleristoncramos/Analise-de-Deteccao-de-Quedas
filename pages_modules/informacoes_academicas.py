"""Página de informações acadêmicas do trabalho, com as logos da UFPI e do CEAD no cabeçalho."""
import base64
import json
from pathlib import Path

import streamlit as st

from pages_modules.navegacao import PAGINA_INICIAL, PAGINA_SISTEMA, ir_para

RAIZ = Path(__file__).resolve().parent.parent
LOGOS = RAIZ / "assets" / "institucional"
DADOS = RAIZ / "data" / "tcc_benchmark_data.json"

TECNOLOGIAS = [
    ("Python 3", "Linguagem de todo o pipeline, da inferência à interface."),
    ("YOLOv11 (Ultralytics)", "Detecção da classe pessoa e rastreamento por ID entre quadros (variantes n, s, m, l e x)."),
    ("OpenCV", "Leitura dos vídeos, redimensionamento e desenho das caixas e alertas."),
    ("NumPy", "Cálculo da razão de aspecto, dos centroides e das velocidades."),
    ("Streamlit", "Interface web com o estúdio de inferência, o dashboard e a auditoria."),
    ("pandas e Plotly", "Organização das métricas do artigo e gráficos interativos."),
    ("fpdf2 e openpyxl", "Exportação dos relatórios de auditoria em PDF e Excel."),
    ("pytest, ruff e GitHub Actions", "Testes automatizados e verificação de estilo a cada envio ao repositório."),
]


def _imagem_base64(caminho: Path) -> str:
    tipo = "image/svg+xml" if caminho.suffix == ".svg" else "image/png"
    return f"data:{tipo};base64,{base64.b64encode(caminho.read_bytes()).decode()}"


def _logo(nome: str, classe: str, alt: str) -> str:
    caminho = LOGOS / nome
    if not caminho.is_file():
        return f'<div class="{classe}" style="border:1px solid #2b2b2b;padding:1.5rem;">{alt}</div>'
    return f'<img class="{classe}" src="{_imagem_base64(caminho)}" alt="{alt}">'


def _metadados() -> dict:
    with open(DADOS, "r", encoding="utf-8") as f:
        return json.load(f)["metadados"]


def _linha(rotulo: str, valor: str) -> str:
    return f"<tr><th>{rotulo}</th><td>{valor}</td></tr>"


def render_informacoes_academicas():
    meta = _metadados()

    st.markdown(f"""
    <div class="cabecalho-academico">
        {_logo("UFPI.svg", "logo-ufpi", "Universidade Federal do Piauí")}
        <div class="centro">
            <div class="linha1">Universidade Federal do Piauí</div>
            <div class="linha2">Centro de Educação Aberta e a Distância (CEAD)<br>
            Curso Superior de Tecnologia em Gestão de Dados (CSTGD)</div>
        </div>
        {_logo("CEAD.png", "logo-cead", "Centro de Educação Aberta e a Distância")}
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    col_voltar, col_sistema, _ = st.columns([1.25, 1, 2.75])
    with col_voltar:
        st.button("Voltar à página inicial", icon=":material/home:", on_click=ir_para,
                  args=(PAGINA_INICIAL,), width="stretch", key="academico_voltar")
    with col_sistema:
        st.button("Get Started", type="primary", icon=":material/arrow_forward:", on_click=ir_para,
                  args=(PAGINA_SISTEMA,), width="stretch", key="academico_sistema")

    st.title("Informações Acadêmicas")
    st.caption("Identificação do Trabalho de Conclusão de Curso que originou este sistema")

    st.subheader("Identificação do trabalho")
    st.markdown('<table class="ficha">' + "".join([
        _linha("Título", meta["titulo"]),
        _linha("Natureza", "Trabalho de Conclusão de Curso, em formato de artigo científico (modelo SBC)"),
        _linha("Tipo de pesquisa", "Prova de Conceito (PoC) aplicada à prevenção de acidentes em ambientes escolares"),
        _linha("Paradigma de avaliação", f"{meta['paradigma_avaliacao']}, com {meta['total_eventos']} eventos contínuos"),
    ]) + "</table>", unsafe_allow_html=True)

    st.subheader("Instituição e curso")
    st.markdown('<table class="ficha">' + "".join([
        _linha("Universidade", "Universidade Federal do Piauí (UFPI)"),
        _linha("Unidade", "Centro de Educação Aberta e a Distância (CEAD)"),
        _linha("Curso", "Curso Superior de Tecnologia em Gestão de Dados (CSTGD)"),
        _linha("Modalidade", "Educação a distância"),
    ]) + "</table>", unsafe_allow_html=True)

    st.subheader("Autoria e orientação")
    st.markdown('<table class="ficha">' + "".join([
        _linha("Autor", ", ".join(meta["autores"])),
        _linha("Orientadores", " e ".join(meta["orientadores"])),
    ]) + "</table>", unsafe_allow_html=True)

    st.subheader("Tecnologias utilizadas")
    st.markdown('<table class="ficha">' + "".join(
        _linha(nome, uso) for nome, uso in TECNOLOGIAS
    ) + "</table>", unsafe_allow_html=True)

    st.subheader("Bases de dados da avaliação")
    st.markdown('<table class="ficha">' + "".join([
        _linha("Dataset simulado (escola)", "Gravado em sala de aula com quedas ensaiadas, oclusões e movimentação em grupo."),
        _linha("UR Fall Detection Dataset", "Benchmark da literatura com 30 quedas e 30 atividades de vida diária."),
        _linha("Vídeos de domínio público", "Cenas reais usadas como teste de estresse de iluminação, ruído e eixo Z."),
    ]) + "</table>", unsafe_allow_html=True)

    st.markdown("""
    <div class="rodape">
        <span>Logotipos: UFPI (Wikimedia Commons, domínio público) e CEAD (site institucional ufpi.br/cead).</span>
        <span>Fontes registradas em assets/institucional/FONTES.md</span>
    </div>
    """, unsafe_allow_html=True)
