# Model Card — The Frequency ML

**Versão:** 1.0  
**Data:** 2026-05-26  
**Autor:** Gabriel Vinicius Nascimento — independent researcher, childhood hepatoblastoma survivor, cisplatin ototoxicity. Built with AI assistance (Claude, Gemini). No institutional affiliation.  
**Status:** Experimental — sem rótulo clínico  

---

## 1. Visão Geral

| Campo | Valor |
|-------|-------|
| **Nome do modelo** | HDBSCAN clustering audiométrico (shape-only) |
| **Tipo** | Clustering não-supervisionado |
| **Tarefa** | Descobrir padrões latentes de perda auditiva em dados populacionais |
| **Dados de treino** | NHANES AUX 1999–Mar2020 (9 ciclos) |
| **N treino** | 7.695 (após filtros) |
| **Features** | 14 limiares brutos (500–8000 Hz, bilateral) |
| **Métricas** | ARI, outlier fraction, Gini importance |
| **Uso pretendido** | Pesquisa + simulação de empatia auditiva (The Frequency) |
| **Uso não recomendado** | Diagnóstico clínico individual |

---

## 2. Dados

### 2.1 Fonte

NHANES (National Health and Nutrition Examination Survey), CDC/NCHS. Pesquisa populacional transversal dos EUA, com audiometria tonal pura por ouvido/frequência.

| Ciclo | Arquivo | N bruto | Frequências |
|-------|---------|---------|-------------|
| 1999–2000 | AUX1.xpt | 1.807 | 500–8000 Hz |
| 2001–2002 | AUX_B.xpt | 2.046 | 500–8000 Hz |
| 2003–2004 | AUX_C.xpt | 1.889 | 500–8000 Hz |
| 2005–2006 | AUX_D.xpt | 3.034 | 500–8000 Hz |
| 2007–2008 | AUX_E.xpt | 1.210 | 500–8000 Hz |
| 2009–2010 | AUX_F.xpt | 2.368 | 500–8000 Hz |
| 2011–2012 | AUX_G.xpt | 4.500 | 500–8000 Hz |
| 2015–2016 | AUX_I.xpt | 4.582 | 500–8000 Hz |
| 2017–Mar2020 | P_AUX.xpt | 5.147 | 500–8000 Hz |
| **Total** | | **26.583** | |

### 2.2 Filtros aplicados

| Filtro | Justificativa | Antes | Depois |
|--------|---------------|-------|--------|
| Idade 20–69 | Remover ciclos com elegibilidade diferente (adolescentes, 70+) | 26.583 | 14.824 |
| Completude ≥10/14 | Garantir dados suficientes por indivíduo | 14.824 | 13.433 |
| ANY25 (≥1 freq >25 dB) | Remover "sol" saudável que engolia densidade | 13.433 | 7.695 |

### 2.3 Variáveis de confundimento conhecidas

| Variável | Impacto | Tratamento |
|----------|---------|------------|
| Idade | Forte (R² ~0.57 em PTA_high) | Filtro 20–69 + row-centering |
| Ciclo | Moderado (Cramér's V ~0.16) | Validação por ciclo (ARI) |
| Sexo | Fraco (Cramér's V ~0.12) | Não controlado (futuro) |
| 666 (no response) | 511 linhas (1.9%) | Política primária: NaN + flag |

---

## 3. Pré-processamento

### 3.1 Tratamento de limiares

| Código | Significado | Tratamento |
|--------|-------------|------------|
| -10 a 120 dB | Válido | Preservado |
| 666 | No response (censura severa) | → NaN + flag |
| 888 | Could not obtain | → NaN |
| Outros | Missing | → NaN |

### 3.2 Row-centering

Para cada indivíduo *i*:

$$\mu_i = \frac{1}{14} \sum_{f \in F} T_{i,f}$$

$$T^{shape}_{i,f} = T_{i,f} - \mu_i$$

Remove o "nível" médio de perda (quanto a pessoa perde em média) e preserva a "forma" da curva (onde a perda é maior/menor).

### 3.3 Scaling

RobustScaler (IQR-based, quantile_range=(25, 75)). Não assume normalidade. Resistente a outliers.

### 3.4 Redução dimensional

PCA com 95% de variância explicada. Resultado: 10 componentes (de 14).

---

## 4. Modelo

### 4.1 Algoritmo

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).

### 4.2 Hiperparâmetros

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| min_cluster_size | 10 | Menor valor que encontra estrutura (grid testado: 5–200) |
| min_samples | 5 | Default do HDBSCAN, testado com 3–20 |
| metric | euclidean | Padrão para dados contínuos |
| cluster_selection_method | eom | Excess of Mass (padrão) |
| core_dist_n_jobs | -1 | Paralelismo |

### 4.3 Grid testado

| min_cluster_size | min_samples | n_clusters | noise_fraction |
|-----------------|-------------|------------|----------------|
| 5 | 3 | 12 | 0.048 |
| 5 | 5 | 4 | 0.088 |
| **10** | **5** | **2** | **0.076** |
| 15 | 5 | 2 | 0.083 |
| 20 | 5 | 0 | 1.000 |
| 30+ | qualquer | 0 | 1.000 |

---

## 5. Resultados

### 5.1 Clusters encontrados

| Cluster | n | % | Descrição geométrica |
|---------|---|---|---------------------|
| 0 | 7.098 | 92,2% | Perda leve-moderada sloping, bilateral relativamente simétrica |
| 1 | 12 | 0,2% | Assimetria unilateral severa (ouvido direito ~80 dB, esquerdo ~16 dB). Direção direita é notável: firearms exposure tipicamente causa perda esquerda em destros (head-shadow effect). Sugere etiologia diferente de ruído ocupacional. |
| Ruído | 585 | 7,6% | Padrões heterogêneos, perda moderada-severa |

### 5.2 Métricas

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| Ruído HDBSCAN | 7,6% | Baixo (era ~90% antes dos filtros) |
| Cross-cycle ARI | 0,27 | Consistência entre ciclos NHANES com elegibilidade diferente. Valor moderado que reflete variação de composição etária entre ciclos, não falha metodológica. Complementa o Bootstrap ARI (0.68) que mede estabilidade dentro da mesma população. |
| Bootstrap ARI (mediano) | 0,68 | Reprodutibilidade dentro de subamostras da mesma população (mede estabilidade interna) |
| Bootstrap ARI (condicional) | 0,60 | Quando clusters aparecem (85% das subamostras) |

> **Nota:** Cross-cycle ARI e Bootstrap ARI são métricas diferentes. O primeiro mede consistência entre populações diferentes (ciclos NHANES); o segundo mede reprodutibilidade dentro da mesma população. Ambos são reportados para transparência.
| RF AUC (cluster 0 vs 1) | 1,0 | Nota descritiva: separação trivialmente perfeita — rótulos derivam das mesmas features (PCA→HDBSCAN), por isso AUC≈1.0 é o resultado esperado e **não constitui evidência de validade** do cluster |

> **Nota sobre AUC=1.0:** O RF surrogate prevê rótulos derivados das mesmas features de entrada, tornando o resultado circular. A métrica serve exclusivamente para ranquear quais frequências dominam a separação (ver §5.3), não para avaliar robustez ou generalização. Ver `outputs/json/rf_surrogate_cv.json` para validação cruzada e LOO.

#### RF surrogate — validação cruzada (v1.1.5)

| Métrica | Resultado | Interpretação |
|---------|-----------|---------------|
| AUC-ROC (CV k=5, médio) | 1,0000 ± 0,0000 | Circular — esperado (ver nota acima) |
| PR-AUC (CV k=5, médio) | 1,0000 ± 0,0000 | Circular — esperado |
| LOO recall (12 positivos) | 9/12 = 75% | 3 positivos com baixa probabilidade quando excluídos do treino (SEQN 12310, 66373, 88806) — sinal de heterogeneidade dentro do Cluster 1 |

> **LOO (leave-one-out nos 12 positivos):** Treina sem cada positivo e verifica se ele é reconhecido. Recall=75% indica que 9/12 membros do Cluster 1 têm padrão suficientemente distinto para ser detectado mesmo fora do treino. Os 3 casos não reconhecidos (prob. < 0,50) podem representar casos limítrofes ou heterogeneidade geométrica dentro do cluster.

### 5.3 Caixa preta (RF surrogate)

Top 7 features discriminativas (todas do ouvido direito):

| Feature | Gini importance |
|---------|----------------|
| thr_R_1000 | 0,2248 |
| thr_R_500 | 0,2203 |
| thr_R_2000 | 0,1453 |
| thr_R_4000 | 0,1175 |
| thr_R_3000 | 0,1174 |
| thr_R_6000 | 0,0711 |
| thr_R_8000 | 0,0427 |

### 5.4 Tinnitus

| Grupo | Taxa | n válido |
|-------|------|----------|
| Cluster 0 | 18,3% | 4.397 |
| Cluster 1 | 50,0% | 8 |

> **Nota:** A taxa de tinnitus do Cluster 1 é baseada em N=8 indivíduos com dados disponíveis. Interpretar como direcionalmente sugestivo, não estatisticamente definitivo.
| Outliers | 38,0% | 308 |

Chi² p<0,001, Cramér's V=0,126.

---

## 6. Análise de Sensibilidade (H11)

| Política | Tratamento 666 | ARI vs nan | Impacto |
|----------|----------------|------------|---------|
| nan (primária) | 666 → NaN + flag | — | Referência |
| cap125 (alternativa) | 666 → 125 dB + flag | 0,9914 | Mínimo |

511 linhas afetadas (1,9%). ARI 0,99 entre políticas → resultado insensível ao tratamento de 666.

### 6.2 ANY25 Filter Sensitivity

| Configuration | N | Clusters | Noise | ARI vs primary |
|---------------|---|----------|-------|----------------|
| With ANY25 (primary) | 7,695 | 2 | 7.6% | — |
| Without ANY25 | 13,433 | 2 | 4.4% | 0.85 |

The ANY25 filter removes the "healthy core" but does not distort the discovered structure. 98.9% of Cluster 0 members and 75% of Cluster 1 members are preserved across filter settings.

### 6.3 OHHR Projection (corrected ingestion)

| OHHR Configuration | N | Noise | Better-ear PTA × DTT-SRT r |
|--------------------|---|-------|-----------|
| Corrected (point→line→audiogram, HTL+AC) | 581 | 61.4% | **0.85** |
| ~~Earlier (mismatched join key)~~ | ~~581~~ | ~~53.0%~~ | ~~0.015 (artifact)~~ |

> **Correção (2026-06):** a ingestão anterior unia `audiogram_point.audiogramlineid` a `audiogram.audiogramid` (chaves de espaços diferentes), casando só 3.433 de 20.538 pontos e misturando orelha, condução óssea e níveis de desconforto no PTA. Isso gerou um r espúrio de 0.015. Com a cadeia correta o r é 0.85 (ver §7.3).

### 6.4 Bootstrap Stability in 4D Space

| Space | N dims | Median ARI | Runs with clusters | SD |
|-------|--------|------------|-------------------|-----|
| 14D (full thresholds) | 10 PCA | 0.68 | 85% | ~0.40 |
| 4D (binaural mean 500/1k/2k/4k) | 4 PCA | **0.74** | **100%** | **0.016** |

The 4-frequency binaural-mean space is *more reproducible at the micro-partition level* (ARI 0.74), but in this space HDBSCAN finds 257 micro-clusters, not the 2-cluster solution — so this ARI does **not** mean the phenotype structure is more stable. It is the space used for the exploratory OHHR projection, which does not validate the 14D phenotypes.



---

## 7. Validação

### 7.1 Validação por ciclo (approximate_predict)

| Ciclo | n teste | ARI |
|-------|---------|-----|
| 1999–2000 | 949 | 0,17 |
| 2001–2002 | 1.031 | 0,21 |
| 2003–2004 | 1.000 | 0,18 |
| 2011–2012 | 2.238 | 0,37 |
| 2015–2016 | 2.477 | 0,41 |

ARI médio: 0,27. Ciclos mais recentes (maior N) têm ARI mais alto.

### 7.2 Bootstrap (100 runs × 80%)

- 85% das subamostras encontraram 2 clusters
- ARI mediano: 0,68
- ARI condicional (quando 2 clusters): 0,60
- 15% de falha: Cluster 1 (12 pessoas) não forma quando subamostrado

### 7.3 Validação externa — OHHR

**Executado.** OHHR (Oldenburg Hearing Health Record; Jafri et al., 2025): 581 adultos (mediana idade 71, PTA mediana 45 dB), CC BY 4.0.

**Pipeline aplicado (espaço comum 4D — distinto do modelo principal 14D):**
1. Extração das 4 frequências comuns (500, 1000, 2000, 4000 Hz); NHANES reduzido a média binaural por frequência (4 variáveis). As 3 freqs só-NHANES (3000/6000/8000) são descartadas, não imputadas.
2. Row-centering (mesma operação).
3. RobustScaler **novo** e PCA **novo** (4 comp, 100% variância trivial) ajustados no NHANES-4D e aplicados ao OHHR. O scaler/PCA 14D do modelo principal **não** são usados (incompatíveis dimensionalmente).
4. HDBSCAN (mcs=10, ms=5) ajustado no NHANES-4D → **257 micro-clusters**, não os 2 do modelo principal.
5. `approximate_predict` projeta o OHHR nesse modelo 4D.

**Resultados (ingestão corrigida 2026-06):**
- 61,4% do OHHR caiu como ruído (vs 37,5% do NHANES no mesmo espaço 4D) — esperado, pois OHHR é mais velho e clínico.
- Correlação melhor-orelha PTA (bruto) × DTT-SRT: **Pearson r=0,85, Spearman r=0,91 (N=581, p<10⁻¹⁶⁰)**. O DTT é fala-no-ruído com ruído fixo (65 dB), onde a audibilidade domina — por isso a correlação é forte.
- **Retratação:** a versão anterior reportava r=0,015 e o interpretava como "Factor D / audiograma não prevê fala". Era artefato de ingestão (chave de merge errada). Neste dataset o audiograma **prevê** o escore de fala; não há dissociação threshold–fala demonstrável aqui.

**Limitação:** o projeto OHHR é um check exploratório de sobreposição de forma, **não** valida o Cluster 0/Cluster 1 (espaço 4D com 257 micro-clusters). OHHR não separa R/L nem cobre 3000/6000/8000 Hz, então não testa a assimetria.

---

## 8. Limitações

### 8.1 Limitações dos dados

1. NHANES é transversal — não há progressão temporal individual.
2. NHANES não tem histórico de cisplatina infantil — "platina-like" é proxy, não confirmação.
3. Frequências limitadas a 500–8000 Hz — ototoxicidade pode começar >8 kHz.
4. Tinnitus é autorrelatado (AUQ191) — disponível apenas em ciclos 2005+.
5. Sem speech-in-noise — NHANES não mede percepção funcional.

### 8.2 Limitações do modelo

1. HDBSCAN é sensível a min_cluster_size — clusters pequenos podem não formar.
2. Row-centering remove nível — não captura "quão ruim" é a perda, apenas a forma.
3. 14 dimensões são poucas — mas capturam 95% da variância.
4. O Cluster 1 (12 pessoas) é muito pequeno para generalização populacional.
5. Os 15% de falha no bootstrap mostram sensibilidade à amostragem.

### 8.3 Limitações éticas

1. Nenhum cluster recebeu rótulo clínico — é geometria, não diagnóstico.
2. O caso pessoal do fundador não foi usado no treino.
3. Prevalências não devem ser inferidas sem survey weights.
4. O modelo não deve ser usado para decisões clínicas individuais.

---

## 9. Uso Recomendado

| ✅ Pode | ❌ Não deve |
|---------|------------|
| Pesquisa em padrões audiométricos | Diagnóstico clínico individual |
| Simulação de empatia auditiva | Inferência prevalência sem pesos |
| Geração de hipóteses clínicas | Substituir audiologista |
| Validação de caso pessoal como ponto externo | Usar dados pessoais como base estatística |

---

## 10. Reprodutibilidade

### 10.1 Ambiente

```
Python 3.13+
numpy, pandas, scipy, scikit-learn, hdbscan, joblib, plotly
```

### 10.2 Scripts

27 scripts Python, numerados sequencialmente. Cada script tem checkpointing (não re-executa se output existe).

### 10.3 Dados

NHANES XPT públicos via CDC. URLs documentadas em `scripts/00_download_nhanes.py`.

### 10.4 Outputs

15+ arquivos JSON com resultados completos. Todos reprodutíveis a partir dos scripts.

---

## 11. Referências

- NHANES: https://wwwn.cdc.gov/nchs/nhanes/
- HDBSCAN: McInnes, L., Healy, J. (2017). Accelerated Hierarchical Density Based Clustering. ICDM 2017.
- ARI: Hubert, L., Arabie, P. (1985). Comparing partitions. Journal of Classification, 2(1), 193-218.

---

## 12. Contato

The Frequency — gabrielviniciusnascimento345@gmail.com

---

*Model Card gerado em 2026-05-26. Nenhum rótulo clínico foi usado no treino.*
