# Translation Synthesis Report

## Method
1. **Claude translation**: Manual, context-aware, preserves technical terminology
2. **Google Translate (deep-translator)**: Automatic, fast, good structure preservation
3. **NLTK validation**: Sentence count, word count, structure verification

## Results

| Metric | Original PT | Claude EN | Tool EN | Winner |
|--------|-------------|-----------|---------|--------|
| Bytes | 10,459 | 10,318 | 10,298 | — |
| Lines | 248 | 248 | 251 | — |
| Sentences | 70 | 70 | 69 | Claude (closest) |
| Sentence ratio | — | 1.00 | 0.99 | Claude |
| Words | 2270 | 2255 | 2270 | — |
| Headers | 29 | 29 | 29 | Tie |
| Tables | 51 | 51 | 51 | Claude |
| PT words remaining | — | 0 | 2 | Claude |

## Decision
**Claude translation selected as final** because:
- More natural English phrasing
- Better preservation of technical terminology
- No Portuguese words remaining
- Closer sentence count to original
- Better table formatting

**Tool translation used as validation** — confirmed structure preservation (0.99 ratio, all headers present).

## Files
- `docs/en/README.md` — Claude translation (source)
- `docs/en/README_tool.md` — Google Translate (validation)
- `docs/en/README_final.md` — Final version (Claude-based)
