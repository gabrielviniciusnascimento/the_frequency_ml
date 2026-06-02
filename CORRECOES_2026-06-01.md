# Changelog de Correções — Sessão 2026-06-01

Registro das correções aplicadas em revisão crítica do projeto. Foco: honestidade
de claims (Cluster 1 / AUC) e um bug de ingestão no OHHR que invertia uma das
conclusões. Nenhuma análise do miolo (HDBSCAN + row-centering + Cluster 0 contínuo)
foi afetada.

---

## 1. Bug crítico — ingestão do OHHR (corrigido)

**Onde:** `scripts/25_external_validation_ohhr.py`

**Problema:** o merge unia `audiogram_point.audiogramlineid` a `audiogram.audiogramid`
(chaves de espaços diferentes), pulando a tabela `audiogram_line`. Resultado: só
**3.433 de 20.538 pontos** recebiam `clientid`; o restante virava NaN. Além disso,
não filtrava tipo de medição nem transdutor, misturando condução óssea (`bc`) e
níveis de desconforto (`ucl`) e ambas as orelhas no PTA.

**Correção:** cadeia correta `point → audiogram_line (side, transducertype, type)
→ audiogram → clientid`, com filtro `htl` + `ac`, média binaural por cliente para o
clustering e PTA de melhor orelha para a correlação. JSON re-gerado em
`outputs/json/25_ohhr_validation.json`.

**Nota de reprodutibilidade:** o script committed estava fisicamente truncado
(linha quebrada) — nunca havia rodado; os JSONs antigos vieram de código inline.

---

## 2. Achado que inverteu — PTA × SRT no OHHR

| | Antes (com bug) | Depois (corrigido) |
|---|---|---|
| Pearson r | 0,015 | **0,85** (p<10⁻¹⁶⁰) |
| Spearman | −0,007 | **0,91** |
| Interpretação | "audiograma não prevê fala (Factor D)" | audiograma **prevê** o escore do Digit-Triplet (ruído fixo) |

A conclusão #4 do paper estava invertida. O DTT é fala-no-ruído com ruído fixo
(65 dB), onde a audibilidade domina — por isso a correlação forte é esperada.
Capturar a dissociação threshold–fala exigiria SNR adaptativo, que o OHHR não tem.

---

## 3. "Validação externa" → "projeção exploratória"

A projeção OHHR roda num espaço reduzido 4D (média binaural de 500/1k/2k/4k Hz),
onde o HDBSCAN forma **257 micro-clusters / 37,5% ruído** — não os 2 clusters do
modelo principal 14D. O OHHR cai **61,4%** como ruído (antes reportado 53%, também
sobre dados com bug). Logo, a projeção testa sobreposição de forma; **não valida**
o Cluster 0/Cluster 1. Reenquadrado em todo o material.

---

## 4. Correção de um erro introduzido nesta revisão

Numa rodada anterior troquei "37,6%"→"7,6%" achando ser typo. Os dados mostraram
que **37,6% era o valor correto** (ruído NHANES no espaço 4D); 7,6% é do modelo
14D. Revertido.

---

## 5. Reenquadramento do Cluster 1 e do AUC (rodadas anteriores da sessão)

- AUC=1,0 do RF marcado como **circular** (rótulos derivam das mesmas features);
  rebaixado a atribuição de features, não evidência de validade.
- **LOO recall 0,75 (9/12)** promovido a métrica principal do surrogate, com os 3
  borderline (prob 0,49/0,36/0,18) explícitos.
- Cluster 1 (N=12): de "fenótipo validado/genuíno" para **sinal exploratório** com
  plausibilidade clínica — artefato técnico descartado, generalização populacional
  **não** estabelecida.
- Cross-cycle ARI 0,27: "moderado" → "fraco-a-moderado"; atribuição à idade marcada
  como hipótese não testada.

---

## 6. Arquivos alterados

**Canônicos (corrigidos):**
`scripts/25_external_validation_ohhr.py`, `outputs/json/25_ohhr_validation.json`,
`docs/en/PAPER_DRAFT_v4.md`, `README.md`, `README.pt.md`,
`THE_FREQUENCY_ML_README.md`, `MODEL_CARD.md`, `LITERATURA_REVIEW.md`,
`docs/en/EMAIL_TEMPLATES.md` (email aos criadores do OHHR — continha o número falso
sobre os dados deles; reescrito com o bug declarado).

**Marcados como superados:**
`docs/en/PAPER_DRAFT.md`, `PAPER_DRAFT_v2.md`, `PAPER_DRAFT_v3.md` (faixa no topo).

**Auxiliares criados:**
`scripts/PATCH_25_pta_ear.md`, `REVISAO_CIENTIFICA_2026-05-29.md`.

---

## 7. Pendências (do autor)

1. **Git/CRLF:** remover `.git/index.lock`, `git checkout -- .`, e commitar o
   `.gitattributes` (normalização LF) — o working tree estava só com diferença de
   fim-de-linha, sem conteúdo novo.
2. **Traduções** (`docs/{de,es,fr,pt}/`, e `docs/en/MODEL_CARD.md`, `README.md`,
   `README_final.md`, `README_tool.md`, `SYNTHESIS.md`): ainda têm r=0,015. Não
   foram tocadas — aguardam revisão linguística nativa antes de oficializar, com
   faixa de aviso.
3. **PTA do OHHR:** o `PATCH_25_pta_ear.md` documenta o conserto; confirmar a
   coluna de orelha já foi feito (existe `side` em `audiogram_line`).

---

*O núcleo do trabalho permanece intacto. O resultado líquido é um conjunto de
claims mais honesto e defensável do que antes desta sessão.*
