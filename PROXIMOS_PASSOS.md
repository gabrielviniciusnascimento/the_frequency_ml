# Próximos passos — the_frequency_ml

**Atualizado:** 2026-06-02 (fim de sessão)
**Manuscrito atual:** `docs/en/PAPER_DRAFT_v5_audit.md` (auditoria de reprodutibilidade)

---

## Onde paramos (estado)

- Bug de ingestão OHHR corrigido (r=0,015 → 0,85), cascata propagada no repo.
- Comparativo HDBSCAN × KMeans × GMM rodado, robustez de BIC resolvida (`scripts/26`).
- Paper v5 (auditoria) **completo**: Abstract, Intro, Métodos (com math do row-centering),
  Resultados (3 tabelas), Discussão (4.1–4.5 + threats-to-validity), Conclusão, Referências.
- **Reprodução independente** no seu PC + ambiente fixado (`requirements-lock.txt`).
- Citações verificadas: Xu, Encina-Llamas, Cruickshanks promovidas; von Gablenz removida.
- README alinhado ao frame de auditoria; faixas no MODEL_CARD e no v4.

## ⚠️ PENDENTE AGORA (antes de tudo amanhã): commitar

Há trabalho não commitado no working tree. Primeiro comando do dia:

```powershell
git add docs/en/PAPER_DRAFT_v5_audit.md README.md MODEL_CARD.md docs/en/PAPER_DRAFT_v4.md
git status   # confira que só esses 4 aparecem
git commit -m "docs: align repo to audit frame (README + banners) + verify quarantined citations"
git push
git log --oneline -1   # confirmar
```
Se travar com `.git/HEAD.lock` ou `index.lock`: `del .git\HEAD.lock` / `del .git\index.lock` e repita.

---

## Roteiro por prioridade

### P1 — Fechar o rigor do manuscrito (alto valor, ~1–2h)
1. **Re-rodar o `26_method_comparison.py` numa passada única** no env `frequency` e colar o
   log no fim do v5 (ou num `outputs/logs/`). Já reproduziu por partes; falta o run limpo
   documentado. (`conda activate frequency` → `python scripts\26_method_comparison.py`)
2. **Conferir páginas/DOIs finais** das refs promovidas (Cruickshanks 29(1):59–67;
   Encina-Llamas marcar como *conference presentation*, VCCA2024).
3. **Ler o paper v5 inteiro de uma sentada** com olhos de revisor — caçar qualquer número
   que não bata com `outputs/json/26_method_comparison.json` e `25_ohhr_validation.json`.

### P2 — Consolidar bibliografia e remover duplicação (médio, ~1h)
4. Unificar as referências do v5 com as do v4 numa lista só (hoje estão separadas).
5. Decidir o que fazer com a Dimitrov: ler o paper e confirmar os números (Jaccard, N) antes
   de citar estatística específica — hoje só citamos a direção, não os valores.

### P3 — Higiene do repositório (baixo risco, faz quando quiser)
6. **Traduções** (`docs/{de,es,fr,pt}`): ainda no frame antigo + r=0,015. NÃO oficializar
   sem revisão nativa (combinado). Pôr faixa de aviso enquanto isso.
7. `scripts/25_*.py`: docstring/`status` ainda dizem "Validação externa" → "projeção
   exploratória" (P2 do VERIFICATION_REPORT).
8. `PITCH_MICROSOFT_AI4A.md`: ainda diz "external validation".
9. `LITERATURA_REVIEW.md`: rodapé diz "18 referências" e tem o frame antigo — atualizar.

### P4 — Decisões maiores (quando a cabeça estiver fresca)
10. **Submissão**: o frame de auditoria combina com *JASA Express Letters*, *Trends in
    Hearing*, ou a *VCCA* (onde Encina-Llamas apresentou — comunidade certa, baixa barreira
    para independente). Avaliar qual.
11. **Contato com grupos**: o e-mail à Hearing4all (OHHR) já está reescrito e honesto. O
    scaffold é a isca para colaboração com quem tem 100k+ audiogramas (Dimitrov, Copenhagen).
12. **Produto** (deixado de lado por escolha sua): simulador com sliders = 1 severidade +
    2–3 forma (eixos PCA). Fica para depois do paper.

---

## Gotchas que custaram tempo hoje (para não repetir)
- **Mount/sandbox lag**: o terminal do agente às vezes vê versão velha de arquivos. A verdade
  é o disco do Windows. Em dúvida, confie no PowerShell.
- **Locks do git** (`HEAD.lock`/`index.lock`): sobram de processos interrompidos. `del` resolve.
- **JSON no Windows**: `open()` assume cp1252 → erro em UTF-8. Sempre `encoding='utf-8'`.
- **CRLF**: o `.gitattributes` já normaliza para LF. Avisos "CRLF will be replaced" são benignos.
- **Env**: use `conda activate frequency` (Python 3.11/3.13 + hdbscan 0.8.44). `python -m pip`
  para garantir o alvo certo.

## Pergunta honesta para amanhã
O paper está rigoroso e defensável. A próxima alavanca real **não** é mais polimento — é
**escolher o destino** (submeter onde? contatar quem?). Cuidado com a tentação de revisar
infinitamente em vez de soltar. Quando reabrir, decida: "hoje eu avanço o conteúdo, ou
escolho para onde isso vai?"

---
*Bons sonhos. Se os tokens acabaram, é só reabrir e mandar "vamos do P1". O mapa está aqui.*
