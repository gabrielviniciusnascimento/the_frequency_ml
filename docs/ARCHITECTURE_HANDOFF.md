# The Frequency — Arquitetura + Handoff

**Data:** 2026-06-13
**Para:** retomada/handoff (você ou outro dev/IA). Leitura de cabeça fria. Companheiro de `docs/VISION_full_circle.md`.

> **TL;DR para dormir:** o motor de áudio binaural **já existe e funciona** (`api/index.html` → `applyHearingLoss`). Você NÃO precisa refazer o áudio. O que está capenga é a **casca** (UX/front), há **dois `index.html` conflitantes**, e a **moldura científica do back está defasada** (fala em "cluster 0/1"; a ciência agora é contínuo + cauda unilateral rara). O trabalho é consolidar, alimentar com dado real e reescrever a casca.

---

## 1. Estado atual (o que existe de verdade)

| Componente | Arquivo | Estado | Veredito |
|---|---|---|---|
| Landing/raiz | `index.html` (raiz, ~21 KB) | Estático, **sem áudio** | **Fraco — descartar/absorver** |
| Awareness screen | `api/index.html` (~26 KB) | **Motor binaural funcional** + amostras sintéticas Web Audio | **Manter o motor, refazer a casca** |
| Motor de áudio | `api/index.html:296-328` (`applyHearingLoss`) | Cadeias biquad por orelha (R=ch0, L=ch1) → `ChannelMerger(2)`; 7 peaking/orelha, Q=1.2 | **Bom — é a fundação** |
| API de projeção | `api/app.py` (FastAPI) | Projeta audiograma → posição PCA, percentil, PTA | Funciona; **moldura cluster 0/1 defasada** |
| Artefatos do modelo | `api/artifacts.json` | scaler + PCA + centroides (da sessão 4) | OK |
| Deploy front | `.github/workflows/deploy-pages.yml` | GitHub Pages | OK |
| Deploy API | `api/render.yaml` | Render free tier | OK |

**Tensões de design já no código (decisões a revisitar):**
- `applyHearingLoss` usa ganho `= max(-(limiar*0.8), -40)` — escala 0,8 e teto −40 dB. Isso **suaviza demais a unilateral**: uma orelha de 80 dB vira só −40 dB, longe de "orelha morta". Foi feito p/ evitar clipping; revisitar para o caso unilateral severo ter impacto real (o `docs/VISION_full_circle.md` usa 1:1, Q=1.4).
- `MASTER_GAIN` fixo — não há calibração ao nível de reprodução do dispositivo.

---

## 2. Arquitetura-alvo (o rebuild)

Princípio: **client-first, estático, sem servidor obrigatório.** A experiência de empatia roda 100% no navegador. A API vira opcional (para quem quer integrar programaticamente).

```
┌─────────────────────────── FRONT (estático, deploy em Pages/edge) ───────────────────────────┐
│  UI / Presentation                                                                            │
│   ├─ Awareness screen (seletor de perfil, play/stop, visualizador de espectro)                │
│   ├─ Sliders do contínuo (severidade + forma) ─── usa PCA inverse                             │
│   ├─ Audiograma plot (R/L) + disclaimer + detecção de fone                                     │
│   └─ Share (Web Share API)                                                                     │
│                                                                                               │
│  audio/binauralLoss.(ts|js)   ← EXTRAIR de api/index.html (já funciona)                        │
│   └─ buildBinauralLoss(ctx, source, profile{left_dbhl,right_dbhl}) → ChannelMerger por orelha  │
│                                                                                               │
│  projection/ (port client-side do api/app.py — opcional mas recomendado)                       │
│   └─ row-center → scaler → PCA → nearest centroid → percentil  (matemática pequena, NumPy-free)│
│                                                                                               │
│  data/                                                                                         │
│   ├─ profiles.json        ← perfis-semente (unilateral = 13 casos reais)                        │
│   ├─ artifacts.json       ← scaler+PCA+centroides (reusar api/artifacts.json)                    │
│   └─ pca_inverse.json     ← params p/ sliders (de session4_pca_scaler_params.json)              │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
            │ (opcional)
┌───────────▼─────────────── BACK (opcional, FastAPI em Render) ───────────────┐
│  api/app.py — POST /api/project, GET /api/clusters,/health  (já existe)        │
│  ⚠ Atualizar a moldura: "main continuum + rare unilateral tail", não 0/1       │
└───────────────────────────────────────────────────────────────────────────────┘
```

**Stack recomendada (leve, alinhada ao ethos):** Vite + TypeScript vanilla **ou** Svelte (bundle pequeno). **Evitar** React/Next pesado. Motor de áudio como módulo TS isolado, testável fora da UI. Por que client-first: a matemática de projeção é pequena (matrizes em `artifacts.json` + 4 operações) — dá para portar para JS puro e **eliminar o backend do caminho crítico** (mesmo argumento do skfreeze). A API fica para integradores.

**O que manter / o que refazer:**
- **Manter:** o grafo de áudio binaural (extrair para `audio/binauralLoss.ts`), as amostras sintéticas Web Audio, `artifacts.json`, os configs de deploy.
- **Refazer:** toda a UI/casca; consolidar os dois `index.html` num front só; reescrever a moldura científica.

---

## 3. O que podemos fazer (capacidades, com o que sustenta cada uma)

| Capacidade | Sustentação | Esforço |
|---|---|---|
| **Simulação binaural verdadeira** (orelha morta de um lado) | Motor já existe (`api/index.html`); perfis reais em `28_ipsative_check.json` | **Baixo** (alimentar + ajustar ganho) |
| Perfis-semente fiéis a dado real | 13 casos: R 68–80 dB / L 11–19 dB (`28_*`) | Baixo |
| **Sliders no contínuo real** (severidade + forma) | PCA params em `session4_pca_scaler_params.json` + `inverse_transform` | Médio |
| Personalização (ouça o seu audiograma) | `api/app.py` projeção já pronta; ou port client-side | Médio |
| Detecção/aviso de fone | Web Audio/AudioContext + heurística | Baixo |
| Cross-sensory (anisometropia, 1 olho borrado) | Dados de visão em `data/processed/vis_feature_matrix.csv` | Alto (nova UI) |

---

## 4. O que NÃO está documentado / gaps (dívida a saldar)

1. **Moldura científica defasada no back.** `api/app.py:166-177` descreve "cluster 0/1" como se fossem subtipos. A ciência atual: **contínuo dominante + cauda unilateral rara**. `cluster 1` = a cauda unilateral real (bom!), mas o texto e o conceito de "percentile_within_cluster" precisam ser reescritos para a moldura de contínuo.
2. **Dois `index.html` conflitantes** (raiz vs `api/`) — fonte de confusão. Consolidar num só.
3. **Perfis não-ancorados.** Os presets atuais do awareness screen foram escolhidos à mão; só o unilateral tem dado rastreável (`28_*`). Documentar a proveniência de cada perfil.
4. **Ganho do áudio suaviza a unilateral** (0,8× e teto −40 dB) — decisão a revisitar para o caso severo ter impacto.
5. **Sem detecção de fone** — o efeito binaural se perde em alto-falante; hoje não há aviso.
6. **Sliders/PCA inverse não implementados** — só descritos em `VISION_full_circle.md`.
7. **Projeção é só server-side** — não há port client; o app depende da API para personalização.
8. **Sem testes do motor de áudio.**
9. **Disclaimer incompleto:** simula audibilidade, não distorção supralimiar/recrutamento/fala-no-ruído/zumbido. Ampliar.
10. **Calibração de nível ausente** — `MASTER_GAIN` fixo; a "perda" percebida depende do volume do dispositivo.

---

## 5. O que emergiu nesta sessão (a ciência a carregar)

Tudo em disco, com SEQN rastreável. Não está no README nem no front:

- **Método cross-system de assimetria** (audição/grip/visão), calibrado por null + métrica adimensional — `scripts/27–30`, `grip_*`, `vis_*`, `cmp_dimensionless_asymmetry.py`.
- **O caso unilateral é real, não artefato** — vive no subespaço de diferença, some no de soma; a média binaural o apaga (13/13→0/13). É a base científica da feature-assinatura. (`27_*`, `28_*`)
- **Contínuo, não subtipos** — mata a UI de "3 perfis", motiva sliders.
- **Visão = controle negativo** (abaixo do null até 4σ) — valida que o aparato não inventa excesso.
- **Integridade como assinatura:** mapeamento anatômico via MGATHAND (grip), sanity de extremos, correção de flag de esforço.
- **Paper v6** (`docs/en/PAPER_DRAFT_v6_crosssystem.md`) + **tabela** (`docs/en/table_crosssystem_asymmetry.tex`) + **figura** (`outputs/dashboards/dimensionless_asymmetry_figure.png`).
- **Ponte produto↔ciência:** `docs/VISION_full_circle.md` (spec da simulação binaural).

---

## 6. Handoff checklist (para amanhã)

**Fundamentos (ordem):**
1. [ ] Decidir stack do front (recomendo Vite + TS ou Svelte). Criar projeto novo; aposentar os dois `index.html`.
2. [ ] **Extrair o motor de áudio** de `api/index.html:296-368` para `audio/binauralLoss.ts` (módulo isolado, testável).
3. [ ] Criar `data/profiles.json` com o `unilateral_right_severe` dos 13 casos (números em `VISION_full_circle.md` Parte 2) + 2 ilustrativos marcados como tal.
4. [ ] Reconstruir a awareness screen em volta do motor: seletor de perfil → play binaural → visualizador.
5. [ ] **Detecção/aviso de fone** antes de tocar binaural.
6. [ ] Revisitar o mapeamento ganho→atenuação (0,8/−40 vs 1:1) para o unilateral severo.

**Ciência ↔ produto:**
7. [ ] Reescrever a moldura do back e do front: "contínuo + cauda unilateral", não "cluster 0/1".
8. [ ] Sliders (severidade + PC1/PC2) via `inverse_transform` (extrair params de `session4_pca_scaler_params.json` → `data/pca_inverse.json`).
9. [ ] (Opcional) Portar a projeção do `api/app.py` para client-side → app 100% estático.

**Qualidade:**
10. [ ] Ampliar o disclaimer (audibilidade vs experiência perceptual completa).
11. [ ] Teste do motor de áudio (paridade do grafo, perfil unilateral → som colapsa p/ um lado).

**Critério de pronto (v2):** ouvinte de fone seleciona o perfil unilateral e percebe o som colapsar para uma orelha — o momento de empatia que espelha o resultado-manchete do paper.
