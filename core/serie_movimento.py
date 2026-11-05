"""Série temporal de razão de aspecto e velocidade por pessoa, para o gráfico do Estúdio.

O gráfico acompanha a "pessoa principal" do vídeo, a que foi observada em mais
quadros (ou a que teve queda registrada), e mostra AR e velocidade ao longo dos
quadros junto com os limiares da heurística.
"""
from collections import defaultdict
from typing import Dict, List, Optional


class SerieMovimento:
    def __init__(self):
        self.series: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: {"frame": [], "ar": [], "v": []})
        self.quedas: Dict[int, List[int]] = defaultdict(list)

    def registrar(self, numero_frame: int, pessoas: List[Dict]):
        for p in pessoas:
            if p.get("id") is None or p.get("largura", 0) <= 0:
                continue
            serie = self.series[p["id"]]
            serie["frame"].append(numero_frame)
            serie["ar"].append(round(p["altura"] / p["largura"], 3))
            serie["v"].append(round(float(p["velocidade"]), 2))

    def registrar_queda(self, numero_frame: int, id_pessoa: Optional[int]):
        if id_pessoa is not None:
            self.quedas[id_pessoa].append(numero_frame)

    def pessoa_principal(self) -> Optional[int]:
        if not self.series:
            return None
        com_queda = [i for i in self.series if self.quedas.get(i)]
        candidatos = com_queda or list(self.series)
        return max(candidatos, key=lambda i: len(self.series[i]["frame"]))

    def dados_principal(self) -> Optional[Dict]:
        """Dicionário serializável (vai para o cache de resultados) com a série da pessoa principal."""
        principal = self.pessoa_principal()
        if principal is None:
            return None
        return {"id": principal, **{k: list(v) for k, v in self.series[principal].items()},
                "quedas": sorted(set(self.quedas.get(principal, [])))}
