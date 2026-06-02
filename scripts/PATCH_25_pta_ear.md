# Patch proposto — script 25, construção do PTA (resposta ao flag Gemini-B)

**Problema:** o PTA do OHHR é calculado a partir de um pivot
(`points.pivot_table(..., aggfunc="first")`, linhas 81-84) que **não fixa orelha**.
Se `audiogram_point` tem mais de uma medição por `clientid`×`frequency`
(orelhas diferentes, repetições), `"first"` escolhe uma arbitrariamente.
Isso pode misturar orelhas no PTA e atenuar artificialmente a correlação
PTA×SRT — exatamente o flanco que o parecer levantou.

**Antes de aplicar:** rodar o diagnóstico abaixo para saber QUE colunas de orelha
existem no OHHR `audiogram_point.json`. Sem isso, não dá pra fixar orelha com honestidade.

## Passo 0 — diagnóstico (rodar primeiro)

```python
import json, pandas as pd
from pathlib import Path
OHHR = Path("data/external/ohhr/data")
pts = pd.DataFrame(json.loads((OHHR / "audiogram_point.json").read_text()))
print("colunas:", list(pts.columns))
print(pts.head(20))
# procurar coluna de lado: 'side', 'ear', 'channel', 'lr', etc.
for c in pts.columns:
    if pts[c].dtype == object:
        print(c, pts[c].dropna().unique()[:8])
# quantas medições por cliente x frequência (se >1, o "first" está escolhendo às cegas)
dup = pts.groupby(["audiogramlineid", "frequency"]).size()
print("máx. medições por linha×freq:", dup.max(), "| %>1:", (dup > 1).mean())
```

## Passo 1 — correção (DEPOIS de identificar a coluna de orelha)

Se existir uma coluna de lado (ex.: `side` com valores tipo `left`/`right`):

```python
# Substituir o bloco de pivot (linhas ~76-84) por:
SIDE_COL = "side"          # <-- ajustar ao nome real achado no Passo 0
pts = points.merge(
    audiogram[["audiogramid", "clientid"]],
    left_on="audiogramlineid", right_on="audiogramid", how="left"
)
pts = pts[pts["frequency"].isin(ohhr_freqs)]

# PTA por orelha, depois melhor orelha (menor PTA = melhor audição)
def ear_wide(side_value):
    sub = pts[pts[SIDE_COL] == side_value]
    w = sub.pivot_table(index="clientid", columns="frequency",
                        values="level", aggfunc="mean")  # mean, não first
    return w

right = ear_wide("right")
left  = ear_wide("left")
pta_r = right.mean(axis=1)
pta_l = left.mean(axis=1)
best_ear_pta = pd.concat([pta_r, pta_l], axis=1).min(axis=1)  # melhor orelha
# Para o espaço de clustering, manter a média binaural (coerente com NHANES):
binaural = pd.concat([right, left]).groupby(level=0).mean()
wide = binaural.reset_index()
wide.columns = ["clientid"] + [freq_map[f] for f in sorted(wide.columns[1:])]
# guardar best_ear_pta para a correlação:
best_pta = best_ear_pta.rename("PTA_best_ear").reset_index()
```

E na seção 5 (correlação), trocar:

```python
# ANTES (linha 154): PTA binaural ambíguo
# df["PTA"] = df[COMMON_FREQ_COLS].mean(axis=1)

# DEPOIS: PTA de melhor orelha, explícito
df = df.merge(best_pta, on="clientid", how="left")
valid_corr = df[["PTA_best_ear", "SRT"]].dropna()
pr = stats.pearsonr(valid_corr["PTA_best_ear"], valid_corr["SRT"])
sr = stats.spearmanr(valid_corr["PTA_best_ear"], valid_corr["SRT"])
```

## Passo 2 — também reportar o range do PTA (restrição de range atenua r)

```python
log.info(f"PTA range: {valid_corr['PTA_best_ear'].min():.0f}–"
         f"{valid_corr['PTA_best_ear'].max():.0f} dB, "
         f"SD={valid_corr['PTA_best_ear'].std():.1f}")
```

## O que esperar

- Se a coluna de orelha **não existir** (OHHR realmente só fornece binaural agregado),
  então o "first" não está misturando orelhas e o flag Gemini-B cai por aí —
  mas aí o texto deve dizer que o audiograma OHHR já é binaural, e o r baixo é
  consistente com DTT (fala-no-ruído). **Documentar isso explicitamente.**
- Se existir e o `aggfunc="first"` estava misturando, recalcular deve **subir** o r
  para a faixa típica de fala-no-ruído (~0.3–0.5). Se subir, atualizar §3.6 com o
  novo número. Se permanecer ~0, é achado real (Factor D), e o texto atual já cobre.

**Não apliquei direto no `25_external_validation_ohhr.py`** porque depende do
resultado do Passo 0 (nome/existência da coluna de orelha), que só você consegue
rodar contra os JSONs do OHHR localmente.
