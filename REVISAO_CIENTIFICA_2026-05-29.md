# Revisão Científica Crítica — The Frequency ML

**Revisor:** análise independente (par cego simulado)
**Data:** 2026-05-29
**Arquivos revisados:** MODEL_CARD.md, LITERATURA_REVIEW.md, README.md (com RESPOSTA_GEMINI_CLUSTER1.md e RESULTADOS_SESSAO5.md como contexto de suporte)
**Escopo:** claims vs. evidência, limitações, referências, Cluster 1 (N=12), AUC=1.0. Não inclui: monetização, estrutura de código, features novas.

---

## Resumo do veredito

O projeto é metodologicamente honesto em vários pontos importantes (seção de limitações explícita, ressalva de N=8 no tinnitus, análise de sensibilidade ao código 666, recusa de rótulo clínico). Porém há **três problemas críticos** onde o claim de destaque ultrapassa a evidência, concentrados em: (1) a comunicação do AUC=1.0, (2) uma contradição numérica interna, e (3) o claim "continuum, não tipos discretos" que é refutado pela própria grid de hiperparâmetros do projeto. A maioria dos problemas se resolve **reescrevendo afirmações, não refazendo análises**.

---

## CRÍTICO

### C1 — AUC=1.0 é circular e está sendo comunicado como evidência de validade

**Trecho (MODEL_CARD §5.2):**
> "| RF AUC (cluster 0 vs 1) | 1,0 | Separação perfeita (classes desbalanceadas) |"

**Trecho (RESPOSTA_GEMINI §3.1, listado como evidência de *robustez* do Cluster 1):**
> "| 4 | Interpretabilidade do RF | AUC=1,0, top 7 features = ouvido direito |"

**Problema:** o Random Forest está prevendo rótulos (cluster 0 vs 1) que foram *derivados das mesmas features de entrada* (limiares → PCA → HDBSCAN). Prever um rótulo a partir das variáveis que o geraram é tautológico — AUC=1.0 é o resultado esperado e **não informa nada** sobre realidade biológica, generalização ou robustez do cluster. Some-se a isso: 12 positivos contra 7.098 negativos, sem menção a train/test split ou validação cruzada; com 12 positivos o sobreajuste é quase garantido, e o AUC sob desbalanceamento extremo é justamente a métrica que mais infla. Usar esse AUC como uma das "4 linhas de evidência de robustez" contra a crítica do Gemini é o uso indevido mais sério no material.

**Correção sugerida:** (a) remover "Separação perfeita" e rebaixar o AUC a nota descritiva: *"O RF separa os clusters trivialmente (AUC≈1.0), o que é esperado já que os rótulos derivam das mesmas features — serve apenas para ranquear quais frequências dominam a separação, não como evidência de validade do cluster."* (b) Reportar, se quiser manter a métrica, precision/recall ou PR-AUC com validação cruzada estratificada e leave-one-out nos 12 positivos. (c) Retirar o AUC da lista de "evidências de robustez" do Cluster 1 — a robustez é sustentada pelos outros 3 pontos (persistência temporal, ausência de 666, insensibilidade à censura), não por ele.

### C2 — Contradição numérica interna na fração de ruído do NHANES

**Trecho (MODEL_CARD §7.3):**
> "53% do OHHR caiu como ruído (vs 37.6% no NHANES)"

**Problema:** todas as outras passagens reportam ruído NHANES = **7,6%** (§5.2 "Ruído HDBSCAN 7,6%"; §4.3 grid mcs=10/ms=5 noise_fraction=0,076; README "585 outliers (7.6% noise)"). O valor "37.6%" não corresponde a nenhuma configuração documentada (sem ANY25 é 4,4%). É uma contradição factual dentro do próprio Model Card, no ponto exato de comparação com a validação externa — um revisor vai marcar isso imediatamente.

**Correção sugerida:** verificar nos JSONs e corrigir para o valor real (provavelmente 7,6%). Se 37,6% se referir a outra coisa (ex.: ruído em algum subconjunto), explicitar a base.

### C3 — "Continuum, não tipos discretos" é refutado pela própria grid do projeto

**Trecho (README, Key findings):**
> "| Population hearing loss is a continuum, not discrete types | HDBSCAN found 2 clusters, not 10+ |"

**Problema:** a grid do próprio Model Card (§4.3) mostra que com `min_cluster_size=5, min_samples=3` o HDBSCAN encontra **12 clusters**, e com `5/5` encontra 4. O resultado "2 clusters" é uma *escolha de hiperparâmetro* (mcs=10), não uma propriedade descoberta dos dados. Além disso o row-centering remove deliberadamente o nível, o que comprime a estrutura. Afirmar como achado que "perda auditiva é um continuum" quando a contagem de clusters varia de 2 a 12 conforme o parâmetro é uma sobre-interpretação. Parthasarathy (2020), corretamente citado, encontrou 6 fenótipos no NHANES com GMM — ou seja, a literatura mais próxima encontra estrutura *discreta* no mesmo dado, o que contradiz diretamente o claim.

**Correção sugerida:** reformular para algo defensável: *"Sob row-centering e min_cluster_size=10, o HDBSCAN resolve a população em 2 agrupamentos de densidade + ruído; configurações mais granulares (mcs=5) fragmentam em até 12. Interpretamos a estrutura de forma como predominantemente contínua nesta escala, sem afirmar ausência de subtipos."* Reconciliar explicitamente com os 6 fenótipos de Parthasarathy.

---

## MODERADO

### M1 — Inferência etiológica sobre o Cluster 1 (N=12) excede a evidência

**Trecho (MODEL_CARD §5.1):**
> "Direção direita é notável: firearms exposure tipicamente causa perda esquerda em destros (head-shadow effect). Sugere etiologia diferente de ruído ocupacional."

**Problema:** não há nenhum dado de etiologia, lateralidade de tiro, ocupação ou destreza para esses 12 indivíduos no NHANES. O argumento do head-shadow é uma plausibilidade clínica, não evidência. O "Sugere" é uma boa atenuação, mas o raciocínio em RESPOSTA_GEMINI §3 torna-se **infalsificável**: se a perda fosse esquerda, confirmaria atiradores destros; sendo direita, é "mais interessante / focal". Qualquer observação é acomodada — isso enfraquece, não fortalece, o argumento. A afirmação "perda esquerda é variável demais para formar cluster" (RESPOSTA_GEMINI §2.3) é asserida sem mostrar a distribuição de assimetria esquerda.

**Correção sugerida:** restringir-se ao que é geométrico e verificável: *"Cluster 1 é uma assinatura geométrica de assimetria unilateral direita severa. Não temos dados etiológicos; a perda unilateral severa é inconsistente com o padrão bilateral típico de ruído ocupacional, mas a causa permanece desconhecida."* Remover o raciocínio de head-shadow como suporte. Apresentar a distribuição de assimetria L vs. R em toda a coorte para sustentar (ou não) a afirmação de que o lado esquerdo "não forma cluster".

### M2 — "30 pessoas" combina um cluster com 18 outliers selecionados post-hoc

**Trecho (README, Key findings):**
> "| 30 people have severe unilateral right-ear asymmetry | Cluster 1 (12) + outlier sub-group (18), across 4 NHANES cycles |"

**Problema:** os 18 não são um cluster independente — são pontos de *ruído* (noise) selecionados depois por semelhança ao Cluster 1. Somá-los para anunciar "30" infla o N de um achado que o próprio bootstrap mostra frágil (Cluster 1 não forma em 15% das subamostras; perde 3/12 membros sem ANY25). Apresentar 30 como número de cabeçalho dá impressão de robustez maior do que a evidência sustenta.

**Correção sugerida:** separar claramente: *"Cluster 1 (N=12) mais um sub-grupo de 18 outliers com assimetria parcial semelhante (N=30 no total), sugerindo um continuum de assimetria em vez de uma categoria."* Não usar "30" como métrica de destaque sem essa qualificação.

### M3 — "26.583 pessoas" como base do achado, quando a clusterização usou 7.695

**Trechos (README):**
> "discovers real patterns of hearing loss in 26,583 people"
> "based on real data from 26,583 people"
**(LITERATURA_REVIEW, positioning statement):** "using 26,583 NHANES audiograms"

**Problema:** 26.583 é o N *bruto ingerido*; os clusters vêm de **7.695** após filtros (idade, completude, ANY25) — 18.888 indivíduos foram removidos antes da modelagem. Repetir 26.583 como a população dos padrões descobertos é enganoso.

**Correção sugerida:** usar a formulação dupla de forma consistente: *"ingeridos 26.583 audiogramas; padrões descobertos em uma coorte analítica de 7.695 após filtros."*

### M4 — Cross-cycle ARI 0,27 reenquadrado como "moderado"

**Trecho (MODEL_CARD §5.2):**
> "| Cross-cycle ARI | 0,27 | ... Valor moderado que reflete variação de composição etária entre ciclos, não falha metodológica. |"

**Problema:** ARI de 0,27 é **baixo** (concordância fraca), não "moderado". Atribuí-lo inteiramente à composição etária é uma hipótese plausível, mas não está demonstrada (não há análise mostrando que controlar idade eleva o ARI). É um spin que suaviza um resultado fraco. A nota explicativa que distingue cross-cycle de bootstrap ARI é boa e deve ficar; o que precisa mudar é o adjetivo e a causalidade não comprovada.

**Correção sugerida:** *"Cross-cycle ARI = 0,27 (concordância fraca a moderada). Hipotetizamos que parte se deve a diferenças de composição etária/elegibilidade entre ciclos, mas não testamos esse controle."* Não afirmar "não é falha metodológica" sem evidência — apenas apresentar o número e a hipótese.

### M5 — "Validação externa — OHHR" promete mais do que entrega

**Trechos (README):** "Validates via ... external projection onto OHHR (N=581)"
**(MODEL_CARD §7.3):** "**Executado.** ... 53% do OHHR caiu como ruído"

**Problema:** chamar isso de "validação externa" das *descobertas* (os clusters, a assimetria unilateral) é generoso. O OHHR não separa orelha D/L — logo **não pode testar** o achado central (Cluster 1 unilateral). 53% caiu como ruído, o que é em si um sinal de transferência fraca, atribuído convenientemente a "OHHR é mais velho e clínico". O único resultado positivo reportado (PTA×SRT r≈0) não valida os clusters — é um achado independente que confirma a literatura de speech-in-noise.

**Correção sugerida:** rebaixar de "validação externa" para *"aplicação externa exploratória"*. Declarar explicitamente: *"O OHHR não permite testar o achado de assimetria (sem separação D/L) e 53% projetou-se como ruído, indicando transferência limitada do modelo entre populações. A correlação PTA×SRT≈0 é consistente com a literatura, mas é um achado separado da estrutura de clusters."*

### M6 — Lista de referências incompleta em relação às citações no texto

**Trecho (LITERATURA_REVIEW, rodapé):** "18 referências principais"

**Problema:** vários trabalhos citados no corpo **não aparecem** na lista numerada de 18: Chang & Chinosornvatana (2010), Knight et al. (2005), Oldenburg et al. (2007), Rybak et al. (2007), Ramirez Camacho et al. (2004). Citações-chave do argumento de assimetria não têm referência completa rastreável: "Jenkins et al.", "Schmidt et al.", "(PMC 3567893)", "(PMC 3567893)" e "Cox & Ford (1995), Chung et al." (em RESPOSTA_GEMINI). Afirmar "18 referências" enquanto múltiplas fontes citadas carecem de entrada bibliográfica é uma fragilidade que um revisor marcará.

**Verificações que fiz:**
- **Parthasarathy et al. (2020), Sci Rep 10:6754** — confere exatamente: GMM, 116.400 registros MEE → 10 tipos, 15.380 NHANES → 6 tipos, 46% não classificáveis. Citação correta. ✅
- **Asimetria por cisplatina** — a literatura confirma viés **para o ouvido ESQUERDO** ("the left ear is slightly but significantly more affected", Ear & Hearing 2008; Schmidt/Münster). A LITERATURA_REVIEW cita isso corretamente ("Schmidt et al... left ear"). **Mas isso cria uma tensão não resolvida:** o Cluster 1 do projeto é **direito**, ou seja, oposto ao viés da cisplatina. A própria revisão (Eixo 3) admite que o padrão de cisplatina "pode estar no continuum do Cluster 0, não como cluster separado" — o que é honesto, mas precisa ser dito com a mesma clareza em qualquer lugar onde o caso pessoal (cisplatina) for associado ao Cluster 1.

**Correção sugerida:** (a) completar a bibliografia com todos os trabalhos citados ou remover as citações sem fonte; reconciliar a contagem "18". (b) Tornar explícito no Model Card e no README que a literatura de cisplatina aponta assimetria **esquerda**, contrária ao Cluster 1 — para evitar que o leitor (ou o autor) conecte intuitivamente o caso pessoal de cisplatina ao Cluster 1 direito.

### M7 — Claims absolutos de ineditismo ("Ninguém...")

**Trechos (LITERATURA_REVIEW, Eixo 1):**
> "Ninguém usou HDBSCAN em audiogramas."
> "Ninguém fez row-centering para isolar forma vs nível."
> "The Frequency faz 5 coisas que a literatura não fez"

**Problema:** afirmações negativas absolutas são quase impossíveis de sustentar e fáceis de refutar com um único contraexemplo — especialmente arriscadas dado que a própria revisão cita uma revisão sistemática (PMC 2025) que admite cobrir métodos variados de ML não-supervisionado.

**Correção sugerida:** atenuar uniformemente para *"até onde encontramos na literatura / not found in our search"*. Padrão em papers revisados por pares.

---

## MENOR

### m1 — Bootstrap ARI 0,68/0,60 chamado de "reprodutibilidade" sem qualificar a fragilidade do Cluster 1
**Trecho (MODEL_CARD §5.2):** "Bootstrap ARI (mediano) | 0,68 | Reprodutibilidade dentro de subamostras". ARI 0,68 é estabilidade *moderada* (não alta). O detalhe importante — 15% das subamostras não formam o Cluster 1 e 25% dos membros somem sem ANY25 — está nas Limitações (bom), mas o cabeçalho no README não qualifica. **Sugestão:** acrescentar "(estabilidade moderada)" e levar a ressalva de fragilidade do Cluster 1 para perto de qualquer destaque dele.

### m2 — "Validação de caso pessoal como ponto externo" — N=1 não é validação
**Trecho (MODEL_CARD §9):** "| Validação de caso pessoal como ponto externo |". Um único audiograma não valida estatisticamente um modelo. O texto já ressalva "não como base estatística" (bom), mas a palavra "validação" é inadequada. **Sugestão:** trocar por "ilustração / posicionamento de caso individual".

### m3 — Datação inconsistente de Sanchez-Lopez
**Trecho (LITERATURA_REVIEW, Eixo 1):** "Sanchez-Lopez et al. | 2018/2020" vs. referência #4 "(2020)". **Sugestão:** unificar o ano.

### m4 — 53% de ruído no OHHR como sinal de transferência (ver M5)
Mesmo ponto de M5, em chave menor: o número 53% deveria ser discutido como limitação de generalização do modelo, não só como "esperado".

---

## Pontos fortes (a preservar)

Para calibragem — estes estão corretos e devem ser mantidos como estão:

- **Ressalva do tinnitus N=8** (MODEL_CARD §5.4): "Interpretar como direcionalmente sugestivo, não estatisticamente definitivo." Exatamente o tom certo.
- **Análise de sensibilidade ao código 666** (ARI 0,99 entre políticas) e a verificação de 0×666 nos 12 do Cluster 1 — evidência genuína e bem documentada.
- **Recusa explícita de rótulo clínico** em todo o material.
- **Seção de Limitações (§8)** é honesta e cobre a maioria dos pontos certos (Cluster 1 pequeno demais para generalização, 15% de falha no bootstrap, ausência de speech-in-noise, cisplatina como proxy).
- **Citação de Parthasarathy (2020)** — precisa e bem posicionada.

---

## Prioridade de correção

1. **C1, C2, C3** — bloqueiam credibilidade; corrigir antes de qualquer submissão ou divulgação.
2. **M1, M2, M3, M5** — onde o README/Model Card "vendem" mais do que os dados sustentam.
3. **M4, M6, M7** — rigor de reporte e bibliografia.
4. **m1–m4** — polimento.

Quase tudo é reescrita de texto, não reanálise. A exceção é C1, onde reportar PR-AUC/CV exigiria rodar o RF surrogate de novo com validação cruzada.

---

### Fontes consultadas para verificação
- Parthasarathy et al. (2020), *Scientific Reports* 10:6754 — https://www.nature.com/articles/s41598-020-63515-5
- Left-Right Asymmetry in Hearing Loss Following Cisplatin Therapy in Children, *Ear and Hearing* (2008) — https://pubmed.ncbi.nlm.nih.gov/18772725/
- Cisplatin-Induced Ototoxicity and the Role of Pharmacogenetic Testing, PMC3567893 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3567893/
