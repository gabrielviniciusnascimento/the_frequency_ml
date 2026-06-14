# HANDOFF — Auditoria pré-commit do The Frequency

## PROMPT (cole/abra isto em uma sessão limpa do Claude Code)

> Você tem acesso completo a este repositório. Leia o estado real antes de agir — pipeline,
> scripts em `scripts/`, resultados em `outputs/json/`, e os drafts `docs/en/PAPER_DRAFT_v6_crosssystem.md`
> e `docs/en/PAPER_DRAFT_v5_audit.md`. Não me peça contexto que o código já contém.
>
> Este arquivo contém um diagnóstico adversário já feito e um plano de 8 tarefas pré-commit.
> Sua função: **verificar cada afirmação do diagnóstico contra o código/JSON, julgar quais
> tarefas destravam mais confiança por unidade de trabalho, e executá-las agora** — começando
> pelo conjunto mínimo defensável. Trabalhe em ordem de severidade/custo, não em ordem de
> numeração se sua verificação indicar melhor. Para cada tarefa que rodar:
>   1. confirme primeiro que o problema existe (cite o número/linha que prova);
>   2. implemente no script certo, com seed e N explícitos;
>   3. produza o artefato de demonstração nomeado (tabela/figura/log em `outputs/`);
>   4. atualize os números correspondentes no draft só se o resultado mudar a afirmação.
>
> Regra de escopo rígida: zero features novas, zero reframe, zero ambição adicional. Uma
> tarefa só vale se (a) tornar uma afirmação já existente defensável sob escrutínio, ou
> (b) demonstrar honestamente o que já está lá. Não suavize nenhum número por causa do
> contexto pessoal do autor. **Não toque no Git** — pare antes de commitar e me mostre os
> artefatos. Se uma afirmação central rachar ao ser testada (ex.: o excesso auditivo morre
> sob o null heterocedástico), diga isso explicitamente e proponha a reescrita honesta da
> manchete em vez de esconder.

---

## Posicionamento honesto (restrição de enquadramento — não suaviza nada abaixo)

Aplicação rigorosa e modesta de auditoria de clustering, feita por quem vive a condição que
estuda (perda auditiva de alta frequência por cisplatina). O valor real é o reframe —
"subtipos audiométricos reportados são segmentação de um contínuo, sensível a algoritmo/seed/
especificação" (prior work conceded, executado com calibração honesta) — mais a dimensão de
acessibilidade. A camada cross-system de v6 é um instrumento promissor, não um achado fechado.
Calibre o esforço para essa verdade: sem inflar, sem diminuir. ANY25 continua sendo seleção
não-pareada; "47:0" continua sendo um sorteio só; 13/13 direita-pior continua contradizendo
Cox & Ford até ser explicado.

---

## Diagnóstico — afirmações a verificar antes de executar

Cada item é falseável contra o repo. Confirme antes de corrigir.

- **D1 — Null de realização única.** A cópula é gerada uma vez, seed fixo.
  `cmp_dimensionless_asymmetry.py:57`, `grip_03_null_model.py:46`, `30_null_model.py:208`
  (`np.random.RandomState(RS).multivariate_normal(...)`, RS=42). Logo "69 vs 0", "47:0",
  "15 vs 0" são uma contagem de cauda de um sorteio — sem envelope, sem p empírico.
- **D2 — Inclusão não-pareada entre sistemas.** Auditivo é filtrado por ANY25
  (`30_null_model.py:266`, `cmp_dimensionless_asymmetry.py:96`: `thr[(thr > 25).any(axis=1)]`);
  grip/visão não têm filtro de anormalidade (`grip_03:84`; `vis_03`/`load_2var`). Contradiz a
  afirmação de v6 §2.3 "same constants... no per-system tuning". O gradiente auditivo≫grip>visão
  está confundido com seleção.
- **D3 — Lado unânime contra o mecanismo citado.** Os 13 casos são 100% direita-pior
  (`29_sanity_check_13.json`, todos `R_minus_L`>0). Cox & Ford 1995 (v6 §4.1) prevê esquerda-pior
  em destros. Contradição não-explicada.
- **D4 — "Cluster" ≠ "modo".** `30_null_model.json` tem `n_contrast_gt_50db=69` (ambos os lados),
  mas o cluster HDBSCAN é 13 (um lado). v5 já registra off-by-one N=12 vs 13.
- **D5 — Null insuficiente para a afirmação.** "Excesso sobre cópula gaussiana" só exclui
  dependência gaussiana; não exclui erro de medição heterocedástico nem mistura de patologias.
  v6 §4.4 concede "excess over second-order structure, not every continuum" — o que contradiz o
  abstract ("genuine rather than a preprocessing artifact").
- **Sobrevive ao escrutínio (não mexer, só preservar):** decomposição soma/diferença
  (`28_ipsative_check.json`: recovery 0.92 no diff bruto, 0.0 no soma) e sanity de coleta
  (`29`, `grip_04`).

---

## Plano — 8 tarefas. Cada uma: Tarefa / Critério / Spec / Demonstração / Esforço.

### 1 — Envelope Monte Carlo do null · barato
- **Tarefa:** regenerar cada cópula B=1000 vezes; trocar todo "X vs 0" por real vs distribuição nula com p.
- **Critério:** p < 0,01 para o auditivo em |z|>4 (reportar p das 12 células). Sobrevive a Bonferroni sobre 3×4 testes.
- **Spec:** loop `RandomState(i)`, i∈[1000,1999], em `make_continuous_14`/`copula_2d`; acumular cauda por |z|∈{2,3,4,5}; p=(#{null≥real}+1)/(B+1); reportar média e percentis 2,5/97,5.
- **Demonstração:** tabela `real | μ_null [IC95%] | p` (sistema×|z|) + 1 figura/sistema (histograma do null, linha vertical do real).

### 2 — Seleção pareada entre sistemas · médio
- **Tarefa:** recalcular o gradiente sob inclusão comparável.
- **Critério:** ordenamento auditivo>grip>visão preservado E ratio auditivo a |z|>3 permanece >2 com o auditivo SEM ANY25. Se colapsa/inverte, era artefato de seleção.
- **Spec:** 3 políticas em `cmp_dimensionless_asymmetry.py` — (i) sem filtro; (ii) anormalidade em todos: auditivo ANY25, grip ≥1 mão <P20 idade×sexo intra-amostra, visão ≥1 olho |SE|≥1,0 D; (iii) atual. Null B=1000 da tarefa 1 em cada.
- **Demonstração:** tabela 3 políticas × 3 sistemas (ratio, p) + legenda dizendo em quais o gradiente sobrevive.

### 3 — Decomposição da cauda por lado · barato
- **Tarefa:** tabular a cauda completa (>50 dB, n=69; e |z|>4) por sinal R-pior vs L-pior; testar vs 50/50.
- **Critério:** reportar fração R-pior com binomial; se >0,65 (p<0,01 em n=69), declarar como propriedade populacional a explicar, não acaso.
- **Spec:** sinal de `PTA_R−PTA_L` na cauda do `30_null_model`; binomial/qui-quadrado vs 0,5; cruzar SEQN extremos com a variável de ordem de teste/condução do arquivo AUX, se presente.
- **Demonstração:** tabela n-por-lado por threshold + log binomial; nota no §3.4/§4.1 reconciliando ou retirando Cox & Ford.

### 4 — Null heterocedástico de medição · médio
- **Tarefa:** segundo null com ruído de medição crescente com o nível; excluir test-retest de orelha pior.
- **Critério:** excesso auditivo a |z|>4 sobrevive com p<0,01 (ratio >5). Se morre, manchete vira artefato de medição.
- **Spec:** somar ruído por canal SD=5+0,1·nível_dB (≤~17 dB a 120 dB, conservador); B=1000; recontar cauda.
- **Demonstração:** coluna lado a lado: `p (cópula gaussiana) | p (heterocedástico)`.

### 5 — Null com dependência de cauda (t-cópula) · médio
- **Tarefa:** repetir contra t-cópula (tem tail dependence; null mais hostil).
- **Critério:** excesso auditivo a |z|>4 sobrevive à t-cópula (ν=4) com p<0,01.
- **Spec:** amostrar t multivariada (ν=4, mesma corr de postos), mapear U via `t.cdf`, mesmas marginais empíricas; B=1000.
- **Demonstração:** 3ª coluna no quadro de nulls (gaussiana | heterocedástico | t-cópula).

### 6 — Estabilidade do cluster N=13 · médio
- **Tarefa:** varrer HDBSCAN; checar se o modo destacado e a ablação de pooling dependem de hiperparâmetro.
- **Critério:** cluster destacado em ≥80% das células, N em ±3, fração R-pior reportada; se não, reescrever a ablação em torno do threshold |z|, não do cluster.
- **Spec:** grid `random_state`∈{0,1,2,7,42} × `min_cluster_size`∈{5,10,15,20} × `min_samples`∈{1,5,10} em `27_binaural_pooling_ablation.py`.
- **Demonstração:** tabela de dispersão (N, survival_rate, %R-pior) sobre o grid.

### 7 — Replicação externa no OHHR · caro
- **Tarefa:** aplicar decomposição soma/diferença + cauda ao audiograma OHHR (único cohort com R/L no repo).
- **Critério:** recovery do subespaço-diferença >0,5 com lado reportado. Qualquer replicação fora do NHANES enfraquece "específico do NHANES".
- **Spec:** reconstruir `audiogram_point→line→audiogram` (type=htl, ac); rodar `28`/`30`; N=581.
- **Demonstração:** linha "OHHR" no quadro de decomposição (recovery, fração R-pior).

### 8 — Controle de sensibilidade informativo · médio
- **Tarefa:** trocar o controle discreto strawman por injeção de assimetria de magnitude conhecida.
- **Critério:** recall ≥0,90 para gaps injetados ≥50 dB.
- **Spec:** injetar k=1% de casos com gap fixo ∈{30,40,50,60} dB; medir recall por magnitude.
- **Demonstração:** curva recall-vs-magnitude.

---

## Ordem e corte

- **Conjunto mínimo defensável: 1 → 2 → 3 → 4.** Fecha inferência sob seleção pareada (1+2),
  o padrão de lado (3) e a objeção de artefato de medição (4). 1 e 3 são baratas — comece por elas.
- **Bom ter (pós-commit): 5, 6, 7, 8.** Endurecem, mas a afirmação central sobrevive sem eles se 1–4 passarem.
- **Cortado (falha no filtro a/b):** nulls adicionais além de 4 e 5; survey weights (não é afirmação
  existente); novos sistemas pareados; melhorias em API/dashboards/traduções. "Seria interessante" ≠ "é necessário".

**Não commitar.** Parar nos artefatos e reportar o que passou, o que rachou, e o que exige reescrita.
