import streamlit as st
import pandas as pd
import io
from fpdf import FPDF
from fpdf.enums import XPos, YPos

# Equivalente ao antigo ln=True do fpdf
NOVA_LINHA = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}


def _texto_pdf(valor) -> str:
    """As fontes padrão do PDF só cobrem Latin-1; troca o resto por '?' em vez de falhar."""
    return str(valor).encode("latin-1", "replace").decode("latin-1")

class PDFRelatorio(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 14)
        self.cell(0, 8, 'UFPI - Centro de Educacao Aberta e a Distancia (CEAD)', **NOVA_LINHA, align='C')
        self.set_font('Helvetica', '', 10)
        self.cell(0, 6, 'Sistema de Analise e Deteccao de Quedas - TCC CSTGD', **NOVA_LINHA, align='C')
        self.cell(0, 6, 'Relatorio de Auditoria e Ocorrencias de Risco', **NOVA_LINHA, align='C')
        self.ln(5)
        self.set_draw_color(200, 200, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.cell(0, 10, f'Pagina {self.page_no()}', align='C')

def gerar_pdf(df: pd.DataFrame, titulo_video: str, total_quedas: int) -> bytes:
    pdf = PDFRelatorio()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Sumario da Sessao de Analise', **NOVA_LINHA)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, _texto_pdf(f'Origem do Video / Fonte: {titulo_video}'), **NOVA_LINHA)
    pdf.cell(0, 6, f'Total de Eventos Registrados: {len(df)}', **NOVA_LINHA)
    pdf.cell(0, 6, f'Total de Quedas Confirmadas: {total_quedas}', **NOVA_LINHA)
    pdf.ln(6)

    pdf.set_font('Helvetica', 'B', 11)
    pdf.cell(0, 8, 'Historico Cronologico de Eventos', **NOVA_LINHA)
    
    # Cabeçalho da tabela
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(20, 7, 'Frame', border=1, fill=True)
    pdf.cell(25, 7, 'Horario', border=1, fill=True)
    pdf.cell(30, 7, 'Tipo', border=1, fill=True)
    pdf.cell(30, 7, 'Postura', border=1, fill=True)
    pdf.cell(30, 7, 'Velocidade', border=1, fill=True)
    pdf.cell(55, 7, 'Detalhes', border=1, fill=True, **NOVA_LINHA)

    # Linhas da tabela
    pdf.set_font('Helvetica', '', 8)
    for _, row in df.iterrows():
        tipo_str = str(row.get('Tipo', ''))
        # Destaque se for QUEDA
        if tipo_str == "QUEDA":
            pdf.set_text_color(200, 0, 0)
        else:
            pdf.set_text_color(0, 0, 0)

        pdf.cell(20, 6, _texto_pdf(row.get('Frame', '')), border=1)
        pdf.cell(25, 6, _texto_pdf(row.get('Timestamp', '')), border=1)
        pdf.cell(30, 6, _texto_pdf(tipo_str), border=1)
        pdf.cell(30, 6, _texto_pdf(row.get('Postura', 'N/A'))[:20], border=1)
        pdf.cell(30, 6, _texto_pdf(row.get('Velocidade', 'N/A'))[:20], border=1)
        pdf.cell(55, 6, _texto_pdf(row.get('Detalhes', ''))[:35], border=1, **NOVA_LINHA)

    pdf.set_text_color(0, 0, 0)
    return bytes(pdf.output())

def render_auditoria_logs():
    st.title("📋 Painel de Auditoria e Logs de Eventos")
    st.caption("Inspeção de ocorrências em tempo real, auditoria de segurança preditiva e exportação de relatórios técnicos")

    eventos = st.session_state.get("historico_eventos", [])

    if not eventos:
        st.info("ℹ️ Nenhum evento registrado na sessão ativa. Execute uma inferência no módulo **Estúdio de Inferência** para gerar ocorrências.")
        return

    df_eventos = pd.DataFrame(eventos)

    # No modo comparativo os dois cenários são registrados; os totais devem ser de um só
    if "Modo" in df_eventos.columns and df_eventos["Modo"].nunique() > 1:
        modo_sel = st.radio("Cenário", sorted(df_eventos["Modo"].unique()), horizontal=True)
        df_eventos = df_eventos[df_eventos["Modo"] == modo_sel]

    # Métricas de Resumo
    col_m1, col_m2, col_m3 = st.columns(3)
    total_quedas = int((df_eventos["Tipo"] == "QUEDA").sum())
    total_prox = int((df_eventos["Tipo"] == "PROXIMIDADE").sum())

    with col_m1:
        st.metric("Total de Ocorrências", len(df_eventos))
    with col_m2:
        st.metric("Quedas Confirmadas", total_quedas, delta="Crítico" if total_quedas > 0 else "Normal", delta_color="inverse")
    with col_m3:
        st.metric("Alertas de Proximidade", total_prox)

    st.divider()

    # Filtros
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        filtro_tipo = st.multiselect(
            "Filtrar por Tipo de Evento",
            options=df_eventos["Tipo"].unique(),
            default=df_eventos["Tipo"].unique()
        )
    with col_f2:
        filtro_busca = st.text_input("Buscar nos detalhes:", "")

    df_filtrado = df_eventos[df_eventos["Tipo"].isin(filtro_tipo)]
    if filtro_busca:
        df_filtrado = df_filtrado[df_filtrado["Detalhes"].str.contains(filtro_busca, case=False, na=False)]

    st.subheader(f"Tabela de Ocorrências ({len(df_filtrado)} registros)")
    st.dataframe(df_filtrado, width="stretch")

    st.divider()

    # Seção de Exportação
    st.subheader("📥 Exportação de Relatórios")
    col_exp1, col_exp2, col_exp3 = st.columns(3)

    # 1. CSV
    csv_bytes = df_filtrado.to_csv(index=False).encode("utf-8")
    with col_exp1:
        st.download_button(
            label="📄 Baixar Relatório CSV",
            data=csv_bytes,
            file_name="relatorio_ocorrencias_tcc.csv",
            mime="text/csv",
            width="stretch"
        )

    # 2. Excel
    buffer_excel = io.BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
        df_filtrado.to_excel(writer, index=False, sheet_name="Ocorrencias")
    with col_exp2:
        st.download_button(
            label="📊 Baixar Planilha Excel",
            data=buffer_excel.getvalue(),
            file_name="relatorio_ocorrencias_tcc.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch"
        )

    # 3. PDF Técnico
    pdf_bytes = gerar_pdf(
        df=df_filtrado,
        titulo_video=", ".join(df_filtrado["Fonte"].unique()) if not df_filtrado.empty else "Analise",
        total_quedas=int((df_filtrado["Tipo"] == "QUEDA").sum())
    )
    with col_exp3:
        st.download_button(
            label="📑 Baixar Relatório PDF Técnico",
            data=pdf_bytes,
            file_name="relatorio_auditoria_tcc_ufpi.pdf",
            mime="application/pdf",
            width="stretch"
        )
