# Pipeline & methods standards audit — The Frequency ML

**Data:** 2026-06-15 · **Escopo:** o pipeline de clustering/sensibilidade (NHANES, espaço de forma).
**Companion:** `meta_analysis.md` (veredito cross-system), `PRODUCT_v2_SPEC.md`, `spinoffs/frente1-methods-note/METHODS_NOTE.md`.

> Objetivo: registrar (a) como o pipeline está organizado, (b) se está num padrão metodológico
> ótimo, e (c) a bateria de existência-de-estrutura que faltava. Alimenta a seção de métodos do paper.

---

## 0. Veredito

O pipeline é **acima da média da literatura aplicada** em integridade (null calibrado, controle
negativo, diagnósticos não-circulares, disk-truth com SEQN). As lacunas eram de **engenharia de
reprodutibilidade** (pré-processamento duplicado, ambiente não fixado, drift script↔artefato) e a
**ausência dos testes canônicos de *existência* de estrutura discreta**. Esta sessão fechou ambas:
um loader único (`scripts/_shape_space.py`), a bateria `31_cluster_tendency.py`, e o congelamento do
ambiente (`requirements.lock` + carimbo de versões em cada JSON).

A tese **"forma do audiograma = contínuo dominante + outliers; sem subtipos discretos bem separados"**
agora é sustentada por **cinco famílias de método convergentes** (ver §3).

---

## 1. Mapa do pipeline

**Dois idiomas de script (cada um consistente internamente):**

- **Numerados `00–31`** (estilo "pipeline"): docstring PT (`Nome/Tarefa/Input/Output/Dependências`),
  `logging`→`outputs/logs/`, checkpoint (pula se output existe), `RANDOM_STATE=42`, parâmetros
  MAIÚSCULOS, JSON com campos honestos `interpretation`/`review_note`.
- **`audit_01–08`, `grip_*`, `vis_*`** (estilo "audit"): `ROOT=Path(__file__).parents[1]`, conciso,
  PCA `svd_solver="full"` (determinístico), figura via `*_figure.py` que lê o JSON.

**Espaço de forma canônico** (agora fonte única em `scripts/_shape_space.py`):
idade 20–69 → completude ≥10/14 → ANY25 (>25 dB em alguma banda) → **row-centering** (isola forma) →
`RobustScaler(25,75)` → `PCA(0.95, svd_solver="full")`. **N=7.695, 10 PCs, var=0.9561.**

---

## 2. Achados de padrão (e o que foi feito)

| # | Achado | Risco | Ação nesta sessão |
|---|---|---|---|
| 1 | Pré-processamento duplicado em `26`, `27`, `audit_06` | divergência silenciosa entre análises | **`scripts/_shape_space.py`** (`load_cohort` + `shape_embed`); os três refatorados para importá-lo |
| 2 | Duas fontes de coorte: CSV+filtros inline vs parquet `model_ready` | N/SEQN podem divergir | loader único como fonte de verdade do CSV; equivalência de embedding verificada (`np.allclose` 14D/7D/solver-default) |
| 3 | `svd_solver` explícito só nos audits | não-determinismo latente | fixado `svd_solver="full"` no loader |
| 4 | `requirements.txt` só com pisos `>=` | sklearn/hdbscan mudam partição entre versões | **`requirements.lock`** (pip freeze) + **`lib_versions` carimbado em cada JSON** |
| 5 | **Drift script↔artefato**: o `26_method_comparison.json` commitado tinha estrutura mais nova que o `.py` commitado | resultado não reproduzível a partir do código | re-execução regenerou o JSON consistente com o código |
| 6 | Bateria de *tendência* incompleta (faltavam Hopkins, dip, OPTICS, dendrograma) | "contínuo" afirmado sem os testes canônicos de existência | **`31_cluster_tendency.py`** + `31b_tendency_figures.py` |
| 7 | `skfreeze` (frente3) existe mas não está plugado no pipeline | ferramenta de reprodutibilidade ociosa | **resolvido**: `32_freeze_pipeline.py` congela o transform canônico (RobustScaler+PCA) num artefato puro-numpy, parity-check 1e-13; `freeze_pipeline` estendido p/ RobustScaler |
| 8 | `METHODS_NOTE.md` usava "Cluster 1 / right-ear / N=12" pré-audit | dívida de integridade (mesma família do fix do produto) | **resolvido**: reconciliado p/ cauda single-ear simétrica, N instável (snapshot 12 / canônico 13 / sweep 9–24), + recomendação de testar *existência* |

**No padrão (forte, mantido):** `RANDOM_STATE` único; logging+checkpoint; campos honestos de revisão;
null calibrado (`30`, `audit_01/04/05`) + controle negativo (`vis_*`); disk-truth com SEQN;
diagnósticos não-circulares (LOPO/LOBO/dual-encoding, `METHODS_NOTE.md`).

---

## 3. Bateria de existência-de-estrutura (resultado convergente)

Todos no mesmo espaço PCA (N=7.695), `RANDOM_STATE=42`. Cinco famílias, nenhuma sustenta subtipos discretos:

| Família | Teste | Onde | Resultado |
|---|---|---|---|
| Partição/verossimilhança | K-means silhouette / Gap; GMM-BIC | `26` | silhouette máx **0.28** @k=2; GMM-BIC sem mínimo interior robusto (só `full`, raso 1.7%, instável k4→k5) |
| Densidade (HDBSCAN) | config principal | `26`/`08` | 2 clusters, **92.2% dominante**, 7.6% ruído |
| Densidade (OPTICS) | reachability + xi | **`31`** | **1 cluster, 0% ruído**, reachability sem vales (CV=0.34) |
| Tendência | Hopkins | **`31`** | H=**0.82** (não-uniforme; *caveat:* não distingue 1 de k modos) |
| Unimodalidade | Hartigan dip | **`31`** | PC1 p=**1.0**, PC2 p=**0.9995** (eixos de clustering unimodais) |
| Hierarquia | Ward cophenetic + gaps | **`31`** | cophenetic **0.40** (não-hierárquico); razão 2º/1º gap=**0.33** (sem 2º k natural) |

**Nota honesta (dip do eixo bruto):** o eixo `PTA_R − PTA_L` rejeita unimodalidade (p≈0), mas com
efeito **minúsculo** (dip=0.023). Com N=7.695 o teste detecta a quantização de 5 dB da audiometria +
uma cauda contínua pesada — **não** um segundo modo bem separado (ver `cluster_tendency_figure.png`,
painel c: pico central único com caudas simétricas). Consistente com a cauda real e *simétrica*
estabelecida no audit, não com um subtipo lateralizado.

**Figura:** `outputs/dashboards/cluster_tendency_figure.png` (reachability · dendrograma · eixo R−L).

---

## 4. Artefatos desta sessão

- `scripts/_shape_space.py` — loader canônico (`load_cohort`, `shape_embed`, `lib_versions`).
- `scripts/31_cluster_tendency.py` → `outputs/json/31_cluster_tendency.json`.
- `scripts/31b_tendency_figures.py` → `outputs/dashboards/cluster_tendency_figure.png`.
- `scripts/26`, `scripts/27`, `scripts/audit_06` — refatorados para o loader + `lib_versions`.
- `scripts/32_freeze_pipeline.py` → `outputs/json/frozen_shape_pipeline.json` (skfreeze plugado; parity 1e-13).
- `scripts/33_qa_prevalence_distribution.py` → prevalência ponderada vs não-ponderada + QA de skew.
- `tests/test_pipeline_contract.py` — guarda de contrato/drift (golden N, PCs, var, partição, parity, sentinelas do 26).
- `requirements.lock` — ambiente exato; `diptest` adicionado a `requirements.txt`.

## 5. Pendências registradas (não bloqueiam)

- ~~Reconciliar `METHODS_NOTE.md`~~ — **feito** (item 8).
- ~~Plugar `skfreeze`~~ — **feito** via `32_freeze_pipeline.py` → `outputs/json/frozen_shape_pipeline.json` (item 7).
- Opcional: mover figuras de `outputs/dashboards/` para `outputs/figures/` (cosmético).
- Nota de revisão: o tamanho da cauda de assimetria na config canônica é **N=13** (27 + artefato congelado; audit_06 ∈ {9,13,24}); o "12" histórico é um snapshot. Qualquer citação deve usar 13 ou a faixa, não 12.

---

## 6. Checklist metodológico (veredito, 2026-06-15)

| Item | Status | Onde / observação |
|---|---|---|
| Gap statistic (Tibshirani) | ✅ feito | `26`, `30`, `grip_02`, `vis_02` |
| AIC | ✅ feito | `26` (`gmm_block`) |
| Silhouette | ✅ feito | `26`, `13`, `27` (`silhouette_samples`) |
| Davies-Bouldin | ✅ feito | `13` (hard labels; para HDBSCAN usa-se DBCV) |
| DBCV | ✅ feito | `08` (`hdbscan.validity_index`) |
| Hartigan dip | ✅ feito | `31` (PC1/PC2 + eixo R−L) |
| Tamanho mínimo de grupo | ✅ feito | `min_cluster_size` (HDBSCAN) |
| Robustez variando min_cluster_size | ✅ feito | `08` grid [20,40,80]; `audit_06` sweep mcs[5,10,15,20]×ms[1,5,10]×rs |
| Score combinado | ✅ feito | `08` `selection_score` (ruído+fragmentação); `13` `choose_best` (silhouette+DB) |
| QA (matriz/NaN/inf) | ✅ feito | `04_qa_report`, `qa_matrix` em `08` |
| QA de distribuição (skew/kurtose) | ✅ feito | `33` (novo) |
| SEQN (rastreabilidade) | ✅ feito | CSVs de assignment + `27/28/29` |
| Skewed data | ✅ tratado | `30` (cópula preserva skew empírico → significância skew-robusta); limiares right-skewed é o *sinal* |
| Box-Cox / Log / power transform | ⛔ não-necessário (documentado) | dB já é log; RobustScaler+row-center; PCs ~simétricos (\|skew\|<1); ver `33.distribution_qa` |
| Ponderada vs não-ponderada | ✅ feito | `33`: prevalência **robusta** a pesos (WTMEC2YR, max Δ=0.26 pp); clustering não-ponderado por desenho |
| Low-confidence subtyping | ◐ parcial / por desenho | `08` grava `membership_probability`+`outlier_score`; LOPO (methods note); moot dado o veredito "contínuo, sem subtipos" |
| Drift relacionado | ✅ mitigado | drift do `26` corrigido; `lib_versions` em cada JSON; `requirements.lock` |
| Unitário de contrato | ✅ feito | `tests/test_pipeline_contract.py` (golden + parity + sentinelas) |
