# RESULTADOS_SESSAO5

**Projeto:** The Frequency × ML  
**Sessão:** 5 — Caixa Preta + Subdivisão + Outliers  
**Data:** 2026-05-26  
**Status:** Executado. Sem rótulo clínico.  

---

## Tarefa 1 — Subdivisão do Cluster 0

**Resultado:** Cluster 0 **não se subdivide** em fenótipos discretos. É um continuum de severidade leve→moderada.

## Tarefa 2 — Análise dos 585 Outliers

**Resultado:** Os outliers **contêm subestrutura**. Sub-grupo de 18 pessoas com assimetria unilateral direita (similar ao Cluster 1).

| Grupo | n | PTA_high_R | PTA_high_L | Assimetria |
|-------|---|------------|------------|------------|
| Cluster 1 | 12 | 78,6 | 15,8 | 61 dB |
| Sub-grupo outliers | 18 | 83,4 | 24,0 | 58 dB |
| **Total unilateral** | **30** | — | — | — |

## Tarefa 3 — RF Surrogate

**AUC-ROC: 1.0**. Top 7 features são **todas do ouvido direito**. O ouvido esquerdo tem importância ~zero.

## Tarefa 4 — Profile do Cluster 1

12 pessoas, idade mediana 46.5, PTA_high_R 78.6 dB, PTA_high_L 15.8 dB, assimetria mediana 61 dB, tinnitus 50%.

## Tarefa 5 — Tinnitus × Clusters

| Grupo | Taxa tinnitus |
|-------|---------------|
| Cluster 0 | 18,3% |
| Outliers | **38,0%** |
| Cluster 1 | 50,0% (n=8) |

Chi² p<0,001.

---

*Sessão 5 executada com sucesso. Nenhum cluster recebeu rótulo clínico.*

---

## Análises Adicionais (pós-Sessão 5)

### Análise 5 — Sensibilidade sem filtro ANY25

HDBSCAN rodado no dataset completo (13.433 pessoas, só filtro idade+completude):

- **ARI entre com/sem ANY25: 0,85** — estrutura altamente consistente
- Cluster 0: 98,9% dos membros se mantêm
- Cluster 1: 75% se mantêm (3 de 12 caíram como ruído sem ANY25)
- Ruído: 87,7% permanece ruído

**Conclusão:** O filtro ANY25 limpa mas não distorce.

### Análise 6 — OHHR com filtro ANY25

OHHR filtrado (N=537 de 581):

- Ruído: 54,0% (vs 53,0% sem filtro)
- PTA×SRT: r=0,018 (vs 0,015 sem filtro)

**Conclusão:** Inconsistência de pipeline resolvida. Resultados virtualmente idênticos.

### Análise 7 — Bootstrap ARI em 4 dimensões

Espaço de 4 frequências binaural média (500/1k/2k/4k Hz):

- **ARI mediano: 0,74** (vs 0,68 em 14D)
- **100% das subamostras** reproduziram clusters (vs 85% em 14D)
- Desvio padrão: 0,016 (muito baixo)

**Conclusão:** O espaço reduzido é **mais estável**. A validação OHHR é fortalecida.
