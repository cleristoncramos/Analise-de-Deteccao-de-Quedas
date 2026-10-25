"""Mantém a aplicação publicada no Streamlit Community Cloud fora da hibernação.

O Streamlit Cloud coloca o app para dormir quando ele fica algumas horas sem
visitas. Uma requisição HTTP simples não conta como visita, porque o app só
registra atividade quando um navegador abre a sessão (WebSocket). Por isso este
script abre a página num Chromium sem interface (Playwright):
- se o app estiver dormindo, clica no botão "Yes, get this app back up!" e
  espera ele voltar;
- se estiver ativo, espera a interface carregar, o que já conta como visita.

Uso: python tools/manter_app_ativo.py [URL]

Sem URL, usa a variável de ambiente APP_URL ou o endereço publicado.
É executado de tempos em tempos pelo workflow .github/workflows/manter-app-ativo.yml.
"""
import os
import sys

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

URL_PADRAO = "https://deteccao-quedas.streamlit.app"
TEXTO_BOTAO_ACORDAR = "get this app back up"
ESPERA_ACORDAR_MS = 5 * 60 * 1000  # o app pode levar alguns minutos para subir de novo
ESPERA_CARREGAR_MS = 90 * 1000


def app_carregado(pagina) -> bool:
    """Espera a interface do Streamlit aparecer (o app roda num iframe dentro da página)."""
    try:
        pagina.frame_locator("iframe").first.locator("[data-testid='stApp']").wait_for(timeout=ESPERA_CARREGAR_MS)
        return True
    except PlaywrightTimeout:
        pass
    # sem iframe (por exemplo, acesso direto ao subdomínio do app)
    try:
        pagina.locator("[data-testid='stApp']").wait_for(timeout=10_000)
        return True
    except PlaywrightTimeout:
        return False


def main() -> int:
    url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("APP_URL") or URL_PADRAO
    with sync_playwright() as p:
        navegador = p.chromium.launch()
        pagina = navegador.new_page()
        pagina.goto(url, wait_until="domcontentloaded", timeout=ESPERA_CARREGAR_MS)
        pagina.wait_for_timeout(5_000)

        botao = pagina.get_by_role("button", name=TEXTO_BOTAO_ACORDAR)
        if botao.count() > 0:
            print("App em hibernação; clicando para acordar.")
            botao.first.click()
            try:
                botao.first.wait_for(state="detached", timeout=ESPERA_ACORDAR_MS)
            except PlaywrightTimeout:
                print("O botão de acordar não sumiu a tempo.")
                navegador.close()
                return 1
            pagina.reload(wait_until="domcontentloaded")

        ok = app_carregado(pagina)
        # permanece um pouco na página para a visita ser registrada
        pagina.wait_for_timeout(20_000)
        navegador.close()

    print("App ativo." if ok else "A interface do app não carregou a tempo.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
