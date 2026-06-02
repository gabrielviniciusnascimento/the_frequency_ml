# Mudança de eixo: de "descoberta de fenótipos" para "auditoria de reprodutibilidade"

**Data:** 2026-06-02
**Manuscrito-alvo:** `docs/en/PAPER_DRAFT_v5_audit.md`

---

## O que mudou

O projeto deixou de ser um paper que **afirma ter descoberto** fenótipos audiométricos
e passou a ser uma **auditoria metodológica**: testa se os "subtipos" reportados na
literatura sobrevivem a mudança de algoritmo, seed e especificação de modelo.

A pergunta central virou:

> *Os subtipos audiométricos da literatura são estrutura real dos dados, ou artefato
> das escolhas do analista (algoritmo, seed, covariância)?*

Resposta empírica, no NHANES (N=7.695, espaço de forma row-centered): **não sobrevivem.**
Silhouette ≤0,28; Gap-ótimo k=2; K-means seed-instável em k=3–5; mínimo interior do
BIC do GMM só sob `full`, raso e migrante (k=4→5); HDBSCAN com um continente de 92,2%
+ outliers raros. Convergente com Allen & Eddins (2010) e Dimitrov et al. (2026).

## Por que essa mudança (o que significa)

1. **Honestidade de prioridade.** O contínuo já era conhecido (Allen & Eddins, 2010;
   e grupos de 2024–2026 com 80–110k audiogramas). Reivindicar a descoberta seria
   atropelado em revisão. Como auditoria, o projeto **concede a prioridade** e reivindica
   o *método* + o *scaffold* — defensável para um pesquisador independente.
2. **Rigor em vez de tamanho de dado.** Não dá para competir com coortes de 100k. Dá
   para competir em reprodutibilidade: bateria multi-algoritmo + multi-especificação na
   mesma amostra, com seed fixa e scaffold aberto.
3. **O bug vira ativo.** O erro de ingestão do OHHR (merge errado, 3.433/20.538 pontos,
   r espúrio de 0,015 → corrigido para 0,85) entra como **estudo de caso** do scaffold:
   prova que a ferramenta audita integridade de dados antes de rodar modelos.

## O que entra neste commit

**Novo manuscrito (completo):**
- `docs/en/PAPER_DRAFT_v5_audit.md` — Abstract, Introdução, Métodos (com a formalização
  matemática do row-centering: projeção ortogonal `P = I − (1/D)·11ᵀ` e o trade-off
  forma vs. severidade), Resultados (3 tabelas), Discussão (4.1–4.5, incl. "Threats to
  validity / o que falsificaria o achado") e Conclusão. Referências com metadados
  **verificados** (Allen & Eddins = PCA+K-means; Parthasarathy = 6754; Dimitrov 2026;
  ERICA/Sorooshyari 2026; Folmer 2017), e um bloco de citações **em quarentena**
  (Xu, Encina-Llamas, von Gablenz, Cruickshanks) ainda não confirmadas.

**Correções P1 (dívida da sessão anterior):**
- `docs/en/PAPER_DRAFT_v4.tex` — r=0,015 → 0,85 nos 4 pontos; títulos "External
  Validation" → "Exploratory Cross-Population Probe"; faixa no topo "AUTO-GENERATED &
  OUT OF DATE — DO NOT RENDER" (o `.tex` ainda carrega outros overclaims antigos do v4).
- `outputs/json/ohhr_any25_validation.json` — flag `INGESTION_BUG: true` + aviso + valor
  corrigido de referência, para o r=0,015 obsoleto não ser reutilizado silenciosamente.

**Relatório de verificação independente:**
- `outputs/VERIFICATION_REPORT.md` — auditoria do Claude Code (rodou no PC do autor):
  confirmou os números do comparativo, o fix do OHHR, e pegou a divergência N=12 vs N=13
  (reconciliada no texto como exemplo da própria instabilidade de fronteira).

## O que NÃO entra / fica pendente

- **Não** foi feito o "Wald→Ward": esse erro não existe no repo (era correção fantasma
  de um dossiê externo).
- **Não** foi importada a bibliografia de um dossiê externo (continha 6704 errado e
  atribuição de "Ward" incorreta a Allen & Eddins).
- Citações em quarentena precisam de verificação antes de ir ao corpo do texto.
- Traduções (`docs/{de,es,fr,pt}`) e READMEs ainda no frame antigo — `CORRECOES §7`.
- Re-execução independente do grid completo `n_init=10` numa passada só (próximo passo).

## Próximos passos (rigor/reprodutibilidade)

1. Re-rodar o `26_method_comparison.py` completo no PC do autor (sem fatiamento) e travar
   números bit-a-bit.
2. Pin de ambiente (`requirements.txt` com versões exatas — sandbox usou sklearn 1.7/py3.10,
   Claude Code usou 1.8/py3.11).
3. Verificar ou remover as citações em quarentena.
4. Alinhar README/título do repo ao frame de auditoria (cosmético, pode esperar).
