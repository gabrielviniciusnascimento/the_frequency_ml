# Post de LinkedIn — The Frequency (rascunho honesto, frame auditado)

> Versão PT (rede local). Há uma versão EN no fim para alcance internacional (grants, pesquisa).
> Tom: humano, direto, sem overclaim. O diferencial é a integridade, não o hype.

---

## Versão PT

Eu perco os agudos desde criança — ototoxicidade por cisplatina, sequela do tratamento de câncer. Aprendi sozinho a programar e passei meses construindo uma coisa que junta as duas pontas da minha vida: ciência de dados e a perda auditiva que eu carrego.

O resultado é o **The Frequency**: um pipeline aberto e reprodutível que pergunta uma coisa simples sobre qualquer medida feita em pares no corpo (as duas orelhas, as duas mãos, os dois olhos) — *existe uma assimetria real entre os lados, ou é artefato da estatística?*

Aplicado a ~8 mil audiogramas públicos (NHANES), ele encontra um sinal real de assimetria entre as orelhas que:
- sobrevive a três modelos nulos diferentes (incluindo um hostil, com dependência de cauda);
- replica num segundo banco, fora do original;
- e **desaparece quando se faz a média das duas orelhas** — que é exatamente o que a maioria das análises faz antes de olhar.

Essa é a frase que move o projeto: **tornar audível a perda que a média apaga.** A PTA de fala esconde os agudos. A média binaural esconde a perda unilateral. O resumo padrão apaga justamente a perda que mais marca a vida de quem a tem.

A parte que mais me orgulha não é um número bonito. É que eu **auditei meu próprio trabalho e recuei**: a primeira versão sugeria uma história de "trauma lateralizado" que os dados, sob escrutínio, não sustentaram. Eu derrubei minha própria manchete e reescrevi tudo. Ciência é isso — e dá pra fazer com rigor mesmo sem crachá institucional.

O próximo passo é um simulador: você coloca o fone e ouve, de verdade, como é uma perda unilateral severa — o som colapsando para um lado. Empatia a partir de dado real.

Tudo aberto, com prova de autoria e código rodável: [link do repositório]

Se você trabalha com **saúde auditiva, acessibilidade, sobreviventes de câncer infantil, ou ciência de dados em saúde** — eu adoraria conversar. Estou aberto a colaboração, freelance e oportunidades.

#Acessibilidade #SaúdeAuditiva #CiênciaDeDados #PesquisaAberta #MachineLearning

---

## Versão EN (alcance internacional)

I've been losing the high frequencies since childhood — cisplatin ototoxicity, a side effect of cancer treatment. I taught myself to code and spent months building something that joins the two ends of my life: data science and the hearing loss I live with.

The result is **The Frequency**: an open, reproducible pipeline that asks a simple question about any measurement made on paired organs (two ears, two hands, two eyes) — *is there a real asymmetry between the sides, or is it a statistical artifact?*

Applied to ~8,000 public audiograms (NHANES), it finds a real inter-ear asymmetry signal that survives three different null models (including a hostile, tail-dependent one), replicates in a second dataset, and **disappears when you average the two ears** — which is exactly what most analyses do first.

The line that drives the project: **make audible the loss that averaging hides.** The speech average hides the high frequencies; binaural averaging hides one-sided loss. The standard summary erases the very loss that shapes a person's life.

What I'm proudest of isn't a pretty number. It's that I **audited my own work and walked a claim back**: an earlier version suggested a "lateralized trauma" story the data didn't support under scrutiny, so I dismantled my own headline and rewrote it. That's what science is — and it can be done rigorously without an institutional badge.

Everything is open, timestamped, and runnable: [repo link]

If you work in **hearing health, accessibility, childhood-cancer survivorship, or health data science**, I'd love to talk — open to collaboration, freelance, and opportunities.

#Accessibility #HearingHealth #DataScience #OpenScience #MachineLearning
