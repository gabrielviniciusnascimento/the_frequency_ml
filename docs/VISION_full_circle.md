# The Frequency — Visão do círculo completo + spec da simulação binaural verdadeira

**Data:** 2026-06-13
**Status:** documento de visão/produto. Conecta a plataforma de empatia original aos achados da auditoria cross-system e especifica a feature-assinatura.

---

## Parte 1 — O círculo

### Onde começou
Plataforma de empatia: qualquer pessoa ouve como é a perda auditiva, baseado em dados reais (NHANES, 26.583 audiogramas). v1 tinha 3 perfis discretos, Web Audio, e uma API de projeção. Raiz pessoal: ototoxicidade por cisplatina.

### O que a ciência fez com isso
Dois pilares do produto v1 caíram — e no mesmo movimento surgiu o v2:

1. **"3 perfis discretos" é falso.** A forma do audiograma é um contínuo (concedido a Allen & Eddins 2010; reproduzido aqui). Vender 3 caixinhas contradiz o próprio achado. → **Trocar por sliders nos eixos reais** (1 severidade + 2–3 forma/PCA).
2. **Simuladores tratam as duas orelhas iguais.** O resultado mais forte da auditoria (`scripts/27_binaural_pooling_ablation.py`) prova que a média binaural **apaga** a experiência mais visceral: a perda unilateral. → **Simulação binaural verdadeira** (por canal), a feature que ninguém faz certo.

### O fio honesto
A perda por cisplatina é bilateral simétrica de alta frequência — geometria oposta ao herói da ciência (assimetria unilateral). Mas as duas são **a perda que o resumo padrão apaga**: a PTA de fala esconde a alta frequência; a média binaural esconde a unilateral. Tese da plataforma, em uma frase:

> **Tornar audível a perda que a média apaga.**

### A ordem de construção (recomendação)
1. **Simulação binaural verdadeira** (Parte 2) — feature-assinatura, mais visceral, barata, re-une ciência e produto.
2. **Sliders** no contínuo real (severidade + forma).
3. **Personalização** via API/skfreeze (ouça o seu audiograma / o de um familiar).
4. **Cross-sensory** (anisometropia com os dados de visão; fraqueza unilateral de grip).

Pacote para grant de acessibilidade (ex.: Microsoft AI4A): empatia que comove (surdez unilateral real) + ciência calibrada por null + tecnologia leve (skfreeze, roda no navegador).

---

## Parte 2 — Spec técnica: simulação binaural verdadeira

### Princípio
Processar **cada orelha de forma independente** com a curva audiométrica daquela orelha. Para um caso unilateral, a esquerda recebe ~0 dB de atenuação e a direita recebe a curva de perda severa — o ouvinte (de fone) experimenta uma orelha quase morta.

> ⚠️ **Requer fones de ouvido.** Em alto-falantes o efeito binaural se perde. A UI deve detectar/avisar.

### Perfis-semente (números reais)

`unilateral_right_severe` é o **audiograma médio dos 13 casos reais** (`outputs/json/28_ipsative_check.json`; PTA R 74,3 dB / L 15,1 dB; contraste 59,2 dB, no topo 0,4% da população). Os outros dois são ilustrativos da forma, não de indivíduos.

```json
{
  "profiles": {
    "unilateral_right_severe": {
      "label": "Perda unilateral severa (direita) — caso real, N=13",
      "right_dbhl": {"500":68, "1000":68, "2000":70, "3000":78, "4000":76, "6000":80, "8000":80},
      "left_dbhl":  {"500":14, "1000":12, "2000":11, "3000":15, "4000":15, "6000":18, "8000":19},
      "source": "28_ipsative_check.json (group mean)"
    },
    "presbycusis_symmetric": {
      "label": "Presbiacusia simétrica (descendente alta-freq)",
      "right_dbhl": {"500":15, "1000":18, "2000":30, "3000":45, "4000":55, "6000":65, "8000":70},
      "left_dbhl":  {"500":15, "1000":18, "2000":30, "3000":45, "4000":55, "6000":65, "8000":70}
    },
    "cisplatin_like_hf": {
      "label": "Ototoxicidade tipo-cisplatina (alta-freq bilateral)",
      "right_dbhl": {"500":5, "1000":5, "2000":10, "3000":25, "4000":45, "6000":70, "8000":85},
      "left_dbhl":  {"500":5, "1000":5, "2000":10, "3000":25, "4000":45, "6000":70, "8000":85}
    }
  }
}
```

### Grafo de áudio (Web Audio API)

Fonte mono ou estéreo → fan-out para duas cadeias independentes → `ChannelMergerNode` (canal 0 = esquerda, canal 1 = direita) → destino.

```
                 ┌─ [EQ banco L: 7 biquads] → gainL ─┐
 source ─────────┤                                   ├─ ChannelMerger(2) → destination
                 └─ [EQ banco R: 7 biquads] → gainR ─┘
```

Cada cadeia = 7 `BiquadFilterNode` em série (`type="peaking"`), um por frequência audiométrica, com **ganho = −atenuação** daquela banda. Atenuação derivada do limiar (ver mapeamento abaixo).

```js
const FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000];
const Q = 1.4;                 // ~1 oitava de largura por banda
const LISTENING_HEADROOM_DB = 0; // calibra ao nível de reprodução (0 = atenuação = limiar)

// threshold (dB HL) -> atenuação (dB) aplicada à banda.
// Modelo honesto e simples: atenua a banda na medida do limiar.
// Para perda profunda isso aproxima inaudibilidade (correto p/ empatia).
function thresholdToAttenuationDb(thr) {
  return Math.max(0, thr - LISTENING_HEADROOM_DB);
}

function buildEarChain(ctx, thresholdsByFreq) {
  const nodes = FREQS.map((f) => {
    const biq = ctx.createBiquadFilter();
    biq.type = "peaking";
    biq.frequency.value = f;
    biq.Q.value = Q;
    biq.gain.value = -thresholdToAttenuationDb(thresholdsByFreq[String(f)] ?? 0);
    return biq;
  });
  for (let i = 0; i < nodes.length - 1; i++) nodes[i].connect(nodes[i + 1]);
  return { input: nodes[0], output: nodes[nodes.length - 1] };
}

function buildBinauralLoss(ctx, source, profile) {
  const L = buildEarChain(ctx, profile.left_dbhl);
  const R = buildEarChain(ctx, profile.right_dbhl);
  const gL = ctx.createGain(), gR = ctx.createGain();
  const merger = ctx.createChannelMerger(2);

  source.connect(L.input); source.connect(R.input);
  L.output.connect(gL); R.output.connect(gR);
  gL.connect(merger, 0, 0);   // canal 0 = esquerda
  gR.connect(merger, 0, 1);   // canal 1 = direita
  merger.connect(ctx.destination);
  return { merger, gL, gR };
}
```

### Mapeamento limiar → atenuação (decisão de design honesta)
- **Modelo base (audibilidade):** atenuação(f) = limiar(f). Simula a perda de *audibilidade* — o que o tom puro mede. É o que o dado sustenta.
- **O que NÃO simula (declarar na UI):** distorção supralimiar, recrutamento, dificuldade de fala-no-ruído, zumbido. Um audiograma é audibilidade, não a experiência perceptual completa. O disclaimer honesto do v1 deve ser mantido e ampliado.
- **Avançado (opcional):** adicionar ruído de banda + leve saturação nas bandas de perda severa para evocar recrutamento/distorção — marcar claramente como "aproximação perceptual", não medida.

### Sliders (transformação 2, depois)
- **Severidade:** offset uniforme somado às 14 bandas (é o nível que o row-centering removeu).
- **Forma (2–3 sliders):** mover ao longo dos eixos PCA. Os parâmetros do scaler/PCA já estão em `outputs/json/session4_pca_scaler_params.json` e `api/artifacts.json` — reconstrua o limiar por banda a partir de (severidade, PC1, PC2[, PC3]) via `inverse_transform`. Isso mantém o explorador no **contínuo real**, sem perfis falsos.

### Integração / reuso
- A projeção (ouça o seu próprio audiograma) reusa `api/app.py` + skfreeze: entra audiograma → projeta → percentil/posição → alimenta o mesmo grafo binaural.
- Mobile-first e offline-after-load (como o v1). O grafo é leve; roda no navegador sem servidor.

### Critério de pronto (feature-assinatura)
Um ouvinte normal, de fone, seleciona `unilateral_right_severe` e percebe nitidamente o som colapsando para a orelha esquerda — a experiência que a média binaural apagaria. Esse é o momento de empatia que ancora a plataforma e espelha o resultado-manchete do paper.
