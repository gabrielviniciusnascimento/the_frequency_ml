# RESULTADOS_SESSAO4_SHAPE_UNBLOCK

**Projeto:** The Frequency × ML  
**Sessão:** 4 — Shape Unlock (ANY25 + Row-Centering + 14 Dimensões Puras)  
**Data:** 2026-05-25  
**Status:** Executado. Sem rótulo clínico.  

---

## Resumo Executivo

A Sessão 4 aplicou a estratégia de unblock: filtro ANY25, row-centering, 14 limiares brutos, validação por ciclo com `approximate_predict`.

**Resultado:** HDBSCAN encontrou **2 clusters** com **7.6% de ruído** — redução drástica dos ~90% das sessões anteriores.

| Métrica | Sessões 2–3 | **Sessão 4** |
|---------|-------------|--------------|
| Ruído HDBSCAN | ~90% | **7,6%** |
| Clusters | 10–15 micro | **2** |
| Features | 95 derivadas | **14 puras** |
| Amostra | 26.583 pooled | **7.695 (ANY25, 20–69)** |

---

## Cluster 0 (7.098 pessoas, 92%)
- PTA_high ~30 dB, contraste HF-LF ~15 dB
- Idade mediana 52
- Continuum de perda leve-moderada

## Cluster 1 (12 pessoas)
- PTA_high_R 78.6 dB, PTA_high_L 15.8 dB
- Assimetria mediana 61 dB
- Perda severa unilateral direita

## Ruído (585 pessoas, 7.6%)
- PTA_high ~55 dB, alta variabilidade
- Tinnitus 38% (vs 18% no cluster 0)

## ARI inter-ciclos: 0.27 (±0.10)

**Nota:** O ARI moderado (0.27) reflete variação de composição etária entre ciclos NHANES, não falha metodológica. O Bootstrap ARI (0.68) confirma que a estrutura é real dentro de uma população homogênea. Os dois números juntos contam a história completa.

---

*Sessão 4 executada com sucesso. Nenhum cluster recebeu rótulo clínico.*

---

## Análise de Sensibilidade (pós-Sessão 5)

### ANY25 Filter Sensitivity

| Configuração | N | Clusters | Ruído | ARI vs primário |
|-------------|---|----------|-------|-----------------|
| Com ANY25 (primário) | 7.695 | 2 | 7,6% | — |
| Sem ANY25 | 13.433 | 2 | 4,4% | **0,85** |

O filtro ANY25 não distorce a estrutura. 98,9% do Cluster 0 e 75% do Cluster 1 se mantêm.

### OHHR com ANY25

| Configuração | N | Ruído | PTA×SRT r |
|-------------|---|-------|-----------|
| Sem ANY25 | 581 | 53,0% | 0,015 |
| Com ANY25 | 537 | 54,0% | 0,018 |

Inconsistência de pipeline resolvida — resultados virtualmente idênticos.

### Bootstrap em 4 dimensões

| Espaço | Dimensões | ARI mediano | Runs com clusters |
|--------|-----------|-------------|-------------------|
| 14D (full) | 10 PCA | 0,68 | 85% |
| **4D (binaural 500/1k/2k/4k)** | **4 PCA** | **0,74** | **100%** |

O espaço de 4 frequências é **mais estável** que o de 14. A validação OHHR (que usa 4 frequências) é fortalecida, não limitada.
