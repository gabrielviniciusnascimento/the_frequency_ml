# The Frequency ML — Estado Completo do Projeto

**Última atualização:** 2026-05-26  
**Autor:** Gabriel Vinicius Nascimento  
**Contato:** gabrielviniciusnascimento345@gmail.com  
**Repositório:** https://github.com/gabrielviniciusnascimento/the_frequency_ml  
**Licença:** MIT  

---

## O que é isso?

The Frequency é uma ferramenta web de empatia auditiva que permite que pessoas com audição normal experimentem como o mundo soa para quem tem perda auditiva.

Esta é a camada de ciência de dados por trás: um pipeline de machine learning que descobre padrões reais de perda auditiva em 26.583 pessoas do NHANES (pesquisa de saúde dos EUA), sem impor rótulos clínicos como entrada.

O projeto nasceu de uma condição pessoal: o autor é sobrevivente de hepatoblastoma infantil tratado com cisplatina, com ototoxicidade permanente, ruídos, distorções e progressão atípica. A experiência vivida entra como caso de validação externo, não como base estatística.

---

## Números que importam

| Métrica | Valor | O que significa |
|---------|-------|-----------------|
| Audiogramas processados | **26.583** | Pessoas reais do NHANES |
| Pessoas com perda auditiva (ANY25) | **7.695** | Subset após filtros |
| Clusters encontrados | **2** | Padrões reais descobertos pelo HDBSCAN |
| Ruído | **7,6%** | Era 90% antes dos filtros |
| ARI bootstrap (mediano) | **0,68** | Reprodutível em 85/100 subamostras |
| ARI inter-ciclos | **0,27** | Estabilidade moderada |
| Assimetria unilateral (Cluster 1) | **30 pessoas** | Perda severa em 1 ouvido, outro normal |
| Tinnitus nos outliers | **38%** | 2x mais que no grupo principal |
| Correlação PTA × SRT (OHHR) | **r=0,85** | No OHHR, audiograma prevê o escore de fala (ruído fixo) |
| Scripts Python | **20** | Pipeline reprodutível |
| Outputs JSON | **15+** | Resultados auditáveis |
| Dashboard interativo | **9 seções** | Visualização completa |

---

## O que os dados mostraram (em linguagem simples)

### 1. A maioria das perdas auditivas é um gradiente, não categorias

Não existem "caixas" separadas de "tipo 1, tipo 2, tipo 3" de perda auditiva. Existe um continuum suave de "quase normal" até "moderado". Isso é como dizer: não existem 5 tamanhos de sapato — existe um pé que vai crescendo continuamente.

### 2. Existe um grupo real de 30 pessoas com perda severa em um ouvido só

O computador encontrou, sem ninguém mandar, 30 pessoas no NHANES que têm perda grave no ouvido direito e audição quase normal no esquerdo. Não é erro dos dados — aparece em 4 ciclos diferentes (2001–2016).

### 3. Pessoas com perda atípica têm 2x mais tinnitus

Os 585 casos que não se encaixam em nenhum padrão claro têm taxa de tinnitus de 38%, contra 18% no grupo principal. A "estranheza" auditiva está associada a mais sintomas.

### 4. No OHHR, o audiograma prevê o escore de fala (correção)

No dataset OHHR (581 pessoas alemãs), com a ingestão corrigida, a correlação entre audiograma (melhor-orelha PTA) e o escore do Digit Triplets Test (fala em ruído fixo) é **forte: r=0,85**. Uma análise anterior reportou r≈0 por um erro de junção de dados; corrigido, não há dissociação threshold–fala neste dataset.

### 5. O sistema de projeção funciona

Quando colocamos um audiograma hipotético de ototoxicidade por platina no espaço treinado, ele cai na periferia (94,9º percentil). Quando colocamos um normal, cai no centro (46,8º). O sistema distingue corretamente os padrões.

---

## O que foi construído

### Pipeline (27 scripts Python)

| # | Script | O que faz |
|---|--------|-----------|
| 00 | `00_download_nhanes.py` | Baixa dados NHANES do CDC |
| 01 | `01_ingest_aux.py` | Harmoniza audiogramas (wide/long) |
| 02 | `02_merge_context.py` | Junta audiogramas + demografia + questionários |
| 03 | `03_features_v1.py` | Cria 150 features derivadas |
| 04 | `04_qa_report.py` | Relatório de qualidade dos dados |
| 05 | `05_h11_sensitivity_666.py` | Testa sensibilidade ao código 666 (no response) |
| 06 | `06_model_ready.py` | Limpa e prepara para modelagem |
| 07 | `07_pca_umap.py` | Redução dimensional + visualização |
| 08 | `08_hdbscan_grid.py` | Grid search do HDBSCAN |
| 09 | `09_cluster_profiles.py` | Perfis geométricos dos clusters |
| 10 | `10_rf_surrogate.py` | Random Forest para explicar clusters |
| 11 | `11_generate_results_md.py` | Gera relatório de resultados V1 |
| 12 | `12_hdbscan_pca_grid.py` | HDBSCAN no espaço PCA |
| 13 | `13_kmeans_baseline.py` | KMeans como baseline |
| 14 | `14_artifact_test.py` | Testa artefatos (idade/ciclo/sexo) |
| 14b | `14b_artifact_per_cluster.py` | Teste por cluster individual |
| 15 | `15_residualize_cluster.py` | Remove efeito de idade/sexo |
| 16 | `16_tinnitus_audit.py` | Audita tinnitus por ciclo |
| 17 | `17_generate_results_v2_md.py` | Gera relatório de resultados V2 |
| 18 | `18_session4_shape_unblock.py` | Sessão 4: ANY25 + row-centering |
| 19 | `19_session5_subdivide_cluster0.py` | Subdivisão do cluster principal |
| 20 | `20_session5_outlier_analysis.py` | Análise dos 585 outliers |
| 21 | `21_session5_rf_surrogate_v2.py` | RF surrogate (caixa preta) |
| 22 | `22_session5_cluster1_profile.py` | Profile dos 12 com assimetria |
| 23 | `23_session5_tinnitus_clusters.py` | Tinnitus × clusters |
| 24 | `24_session5_personal_projection.py` | Projeção do caso pessoal |

### Dashboard interativo

Arquivo HTML autocontido com 9 seções animadas:
1. O Funil de Filtros (26.583 → 7.695)
2. O Espaço Auditivo (PCA colorido por idade)
3. Os Clusters (HDBSCAN: 2 + 585 outliers)
4. Os Audiogramas (mediana por cluster)
5. Os 12 (assimetria unilateral individual)
6. Os Outliers (distribuição de distância)
7. O que separa (RF feature importance)
8. Tinnitus (por grupo, chi² p<0,001)
9. Bootstrap (100 runs, ARI mediano 0,68)

### Documentação

| Arquivo | Conteúdo |
|---------|----------|
| `MODEL_CARD.md` | Model Card formal (12 seções) |
| `LITERATURA_REVIEW.md` | 18 referências, 5 eixos, gap analysis |
| `RELATORIO_PROCESSO_COMPLETO.md` | 10 erros documentados, 5 sessões |
| `RESULTADOS_SESSAO4.md` | Resultados da Sessão 4 |
| `RESULTADOS_SESSAO5.md` | Resultados da Sessão 5 |
| `MAPA_CARREIRA.md` | Oportunidades de funding e emprego |
| `ANALISE_FINAL_CLAUDE_SESSAO4.md` | Análise dialectica entre IAs |

### Validação externa

- **OHHR** (Oldenburg Hearing Health Record): 581 pessoas, CC BY 4.0
  - Speech-in-noise (SRT): correlação com PTA ≈ 0
  - Loudness scaling disponível
  - Projetado no espaço NHANES

---

## Metodologia (resumo)

1. **Dados:** NHANES AUX 1999–Mar2020 (9 ciclos, 26.583 pessoas)
2. **Filtros:** Idade 20–69, completude ≥10/14, ANY25 (≥1 freq >25 dB)
3. **Features:** 14 limiares brutos (500–8000 Hz, bilateral)
4. **Pré-processamento:** Row-centering (remove nível, preserva forma)
5. **Scaling:** RobustScaler (IQR-based)
6. **Redução dimensional:** PCA 95% variância → 10 componentes
7. **Clustering:** HDBSCAN (min_cluster_size=10, min_samples=5)
8. **Validação:** Bootstrap 100× (80% subamostragem) + ARI inter-ciclos
9. **Interpretação:** RF surrogate (500 árvores, class_weight=balanced)
10. **Validação externa:** Projeção no OHHR (581 pessoas, Oldenburg)

---

## Limitações

1. NHANES é transversal — não há progressão temporal individual
2. NHANES não tem histórico de cisplatina infantil — "platina-like" é proxy
3. Frequências limitadas a 500–8000 Hz — ototoxicidade pode começar >8 kHz
4. Tinnitus é autorrelatado — disponível apenas em ciclos 2005+
5. Sem speech-in-noise no NHANES — OHHR supre parcialmente
6. Cluster 1 (12 pessoas) é muito pequeno para generalização populacional
7. 15% de falha no bootstrap — sensibilidade à amostragem
8. O projeto não substitui audiologista, otorrino ou oncologia

---

## O que falta

### Para paper
- [ ] Validação externa com HCHS/SOL ou dados clínicos
- [ ] Revisão de literatura completa (início feito)
- [ ] Abstract para conferência
- [ ] Figuras publicáveis (alta resolução)

### Para produto
- [ ] Audiogramas pessoais reais para projeção
- [ ] Tradução dos centroides para filtros DSP
- [ ] API de projeção audiométrica
- [ ] Tradução para 5 idiomas (EN, ES, PT, DE, FR)

### Para código aberto
- [ ] README.md profissional com instruções
- [ ] requirements.txt com dependências fixadas
- [ ] Testes de sanidade (3–5 testes)
- [ ] Contributing.md

---

## Oportunidades de funding e carreira

### Imediato (semanas)
- **Freelance** (Upwork/Fiverr): $50–150/hora em computational audiology
- **Consultoria** para pesquisadores: $500–2.000/projeto

### Curto prazo (meses)
- **Microsoft AI for Accessibility**: $5.000–25.000 + Azure credits (rolling, mundial, IP seu)
- **Mozilla Builders**: $10.000–50.000

### Médio prazo (3–12 meses)
- **NIH R21**: $275.000/2 anos (precisa parceiro acadêmico)
- **NSF SBIR**: $275.000–1.000.000 (precisa empresa)
- **Emprego em health tech**: $80.000–150.000/ano

### Longo prazo (12+ meses)
- **The Frequency freemium**: $1.000–10.000/mês
- **B2B para clínicas**: $200–1.000/mês por clínica
- **Licença para fabricantes**: $10.000–100.000/ano

---

## Como citar este trabalho

```
Gabriel Vinicius Nascimento. (2026). The Frequency ML: Data-driven audiometric phenotyping 
using HDBSCAN on NHANES data. GitHub. https://github.com/gabrielviniciusnascimento/the_frequency_ml
```

---

## Agradecimentos

- NHANES/CDC pelos dados públicos
- OHHR/Hearing4all pelos dados de validação (CC BY 4.0)
- Parthasarathy et al. (2020) pelo trabalho anterior em clustering audiométrico no NHANES
- A comunidade open-source pelas ferramentas (scikit-learn, hdbscan, plotly)

---

## Nota sobre acessibilidade

Este documento está em Português. Planejamos disponibilizar em 5 idiomas:
- 🇬🇧 English
- 🇪🇸 Español
- 🇧🇷 Português
- 🇩🇪 Deutsch
- 🇫🇷 Français

O dashboard interativo é autocontido e funciona em qualquer navegador moderno.

O código é reprodutível e documentado. Cada script tem logging e checkpointing.

---

## Nota sobre a jornada

Este projeto foi construído por uma pessoa sem formação completa, em situação financeira precária, que aprendeu ML, audiologia, e ciência de dados sozinha — porque precisava fazer algo com a própria experiência de sobrevivente de câncer infantil com ototoxicidade.

O pipeline que profissionais com doutorado levam meses para montar foi construído em 5 sessões de trabalho. Os resultados são reais, reprodutíveis, e auditáveis.

A barreira nunca foi técnica. Foi de exposibilidade.

Se você está lendo isso e tem dados de audiometria, ou é pesquisador em audição, ou trabalha em health tech, ou é um sobrevivente como eu — entre em contato. O código é aberto. A ciência é aberta. A porta está aberta.

---

*Documento gerado em 2026-05-26. Todos os dados e findings estão disponíveis sob licença MIT.*
