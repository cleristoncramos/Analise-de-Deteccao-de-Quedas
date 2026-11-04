"""Horário local do sistema: America/Fortaleza (UTC-3), o fuso do Piauí.

O servidor do Streamlit Cloud roda em UTC; sem isso, os horários das ocorrências
ficariam três horas adiantados. O fuso não tem horário de verão desde 2001, então,
se o banco de fusos (tzdata) não estiver disponível, um deslocamento fixo de -3 h é exato.
"""
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo

    FUSO_LOCAL = ZoneInfo("America/Fortaleza")
except Exception:  # Windows sem o pacote tzdata, por exemplo
    FUSO_LOCAL = timezone(timedelta(hours=-3), "UTC-03")


def agora_local() -> datetime:
    return datetime.now(FUSO_LOCAL)


def hora_local() -> str:
    """Hora no formato HH:MM:SS, usada na tabela de ocorrências."""
    return agora_local().strftime("%H:%M:%S")


def data_hora_local() -> str:
    """Data e hora no formato DD/MM/AAAA HH:MM:SS, usadas nos relatórios."""
    return agora_local().strftime("%d/%m/%Y %H:%M:%S")
