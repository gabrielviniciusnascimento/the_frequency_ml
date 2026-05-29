# Dialectica: Ferramentas para Submissão arXiv

**Data:** 2026-05-27  
**Contexto:** Paper v4 em Markdown → precisa virar PDF formatado para arXiv  
**Autor:** Gabriel Vinicius Nascimento (independent researcher, sem experiência LaTeX)

---

## O problema

O paper v4 está em Markdown (`PAPER_DRAFT_v4.md`). O arXiv aceita PDF (gerado de LaTeX) ou HTML. Precisamos converter o Markdown em algo submissível.

---

## Ferramentas encontradas

### 1. Pandoc (conversor universal)

| Aspecto | Detalhe |
|---------|---------|
| **O que é** | Conversor universal de formatos. Markdown → LaTeX → PDF |
| **Custo** | $0 (open source) |
| **Instalação** | `apt install pandoc` ou download |
| **Comando** | `pandoc paper.md -o paper.pdf --bibliography=refs.bib` |
| **Prós** | Simples, rápido, universal, suporta citações, tabelas, matemática |
| **Contras** | Templates básicos (precisa customizar para ficar "científico"), tabelas complexas podem quebrar |
| **arXiv?** | Gera .tex intermediário que pode ser submetido direto |
| **Fit para você** | 🟢 **Melhor opção** — mais simples, mais rápido |

### 2. Rxiv-Maker (engine para arXiv)

| Aspecto | Detalhe |
|---------|---------|
| **O que é** | Framework que gera PDFs prontos para arXiv a partir de Markdown |
| **Custo** | $0 (open source) |
| **Instalação** | `pip install rxiv-maker` |
| **Comando** | `rxiv pdf` (dentro do projeto) |
| **Prós** | Feito especificamente para arXiv, suporta figuras programáticas, citações automáticas, validação |
| **Contras** | Mais complexo de configurar, menos maduro que Pandoc |
| **arXiv?** | Sim, é o caso de uso principal |
| **Fit para você** | 🟡 Bom, mas mais setup que Pandoc |

### 3. Overleaf (editor web)

| Aspecto | Detalhe |
|---------|---------|
| **O que é** | Editor LaTeX online com colaboração em tempo real |
| **Custo** | Free tier disponível |
| **Instalação** | Nenhuma (web) |
| **Uso** | Colar texto, usar template de journal/conferência |
| **Prós** | Zero instalação, templates prontos, preview em tempo real |
| **Contras** | Precisa aprender LaTeX básico, não é automatizado |
| **arXiv?** | Exporta .tex e PDF |
| **Fit para você** | 🟡 Bom se quiser controle manual |

### 4. Quarto (publicação científica)

| Aspecto | Detalhe |
|---------|---------|
| **O que é** | Sistema de publicação científica multi-formato |
| **Custo** | $0 (open source) |
| **Instalação** | `pip install quartodoc` ou download |
| **Comando** | `quarto render paper.qmd` |
| **Prós** | Suporta Python/R inline, múltiplos formatos (PDF, HTML, Word), foco científico |
| **Contras** | Mais pesado, curva de aprendizado |
| **arXiv?** | Sim, via LaTeX intermediário |
| **Fit para você** | 🟡 Bom se quiser figuras programáticas |

### 5. Typst (typesetter moderno)

| Aspecto | Detalhe |
|---------|---------|
| **O que é** | Alternativa moderna ao LaTeX |
| **Custo** | $0 (open source) |
| **Instalação** | `cargo install typst` ou download |
| **Prós** | Compilação rápida, sintaxe moderna, crescente |
| **Contras** | Não é padrão arXiv, menos templates, ecossistema menor |
| **arXiv?** | Não nativamente (precisa converter) |
| **Fit para você** | 🔴 Não recomendado para arXiv agora |

### 6. LaTeX direto

| Aspecto | Detalhe |
|---------|---------|
| **O que é** | Linguagem de typesetting padrão acadêmico |
| **Custo** | $0 (TeX Live, MiKTeX) |
| **Prós** | Controle total, padrão arXiv, templates de conferência |
| **Contras** | Curva de aprendizado íngreme, verboso |
| **arXiv?** | Sim, é o formato nativo |
| **Fit para você** | 🔴 Não recomendado como ponto de partida |

---

## Comparação direta

| Critério | Pandoc | Rxiv-Maker | Overleaf | Quarto | Typst | LaTeX |
|----------|--------|------------|----------|--------|-------|-------|
| Custo | $0 | $0 | Free/$ | $0 | $0 | $0 |
| Instalação | Fácil | Média | Nenhuma | Média | Média | Pesada |
| Curva aprendizado | Baixa | Média | Média | Média | Baixa | Alta |
| Markdown → PDF | ✅ | ✅ | ❌ (LaTeX) | ✅ | ❌ | ❌ |
| arXiv ready | ✅ (.tex) | ✅ (.tex) | ✅ (.tex) | ✅ (.tex) | ⚠️ | ✅ |
| Citações | ✅ | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| Tabelas | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Figuras inline | ⚠️ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Automação | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Fit para você | 🟢 | 🟡 | 🟡 | 🟡 | 🔴 | 🔴 |

---

## Veredito

### Caminho recomendado: Pandoc

**Por quê:**
1. Você já tem o paper em Markdown
2. Pandoc converte Markdown → LaTeX → PDF em um comando
3. Não precisa aprender LaTeX
4. Suporta citações (`.bib`), tabelas, matemática
5. Gera `.tex` intermediário que pode ser submetido no arXiv
6. É a ferramenta mais madura e confiável

### Comando para gerar o PDF:

```bash
pandoc docs/en/PAPER_DRAFT_v4.md \
  -o paper.pdf \
  --bibliography=refs.bib \
  --template=arxiv-template.tex \
  --pdf-engine=xelatex
```

### Para gerar o .tex (submissão arXiv):

```bash
pandoc docs/en/PAPER_DRAFT_v4.md \
  -o paper.tex \
  --bibliography=refs.bib \
  --template=arxiv-template.tex
```

### Caminho alternativo: Rxiv-Maker

Se quiser algo mais automatizado e com figuras programáticas:
```bash
pip install rxiv-maker
rxiv init my_paper
# Copiar conteúdo do paper v4 para o projeto
rxiv pdf
```

---

## O que eu faria no seu lugar

1. **Pandoc** para gerar o PDF rapidamente
2. **Overleaf** para ajustes finais (se precisar)
3. **arXiv upload** direto do .tex gerado

Quer que eu instale o Pandoc e gere o PDF do paper v4?
