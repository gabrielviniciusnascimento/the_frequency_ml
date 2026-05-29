# Arquitetura — The Frequency Audio Engine v2

**Data:** 2026-05-28  
**Objetivo:** Substituir os sine waves sintéticos por simulação realista de perda auditiva  
**Princípio:** Separar claramente o que cada camada faz para evitar os erros de iteração passada

---

## Problema atual

O `index.html` atual tem 3 problemas técnicos:

1. **Samples são gerados por oscillators** — soam como buzinas, não como fala/música/ruído
2. **Simulação usa peaking filters** — não captura recruitment, distorção, tinnitus
3. **Tudo num único arquivo HTML** — difícil de testar, debugar, ou substituir componentes

---

## Arquitetura proposta (3 camadas)

```
┌─────────────────────────────────────────────┐
│  CAMADA 1: ÁUDIO (samples reais)            │
│  - Conversa CC0 / gerada por TTS            │
│  - Música CC0                               │
│  - Ruído urbano CC0                         │
│  - Formato: WAV/MP3, mono, 44.1kHz          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  CAMADA 2: MOTOR DE SIMULAÇÃO              │
│  - Opção A: 3DTI Toolkit JS (preferencial) │
│  - Opção B: Web Audio API nativo (fallback) │
│  - Input: áudio + 14 thresholds             │
│  - Output: áudio processado                 │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  CAMADA 3: UI + VISUALIZAÇÃO               │
│  - Seleção de perfil                        │
│  - Visualizer (canvas)                      │
│  - Toggle normal/perda                      │
│  - Share buttons                            │
└─────────────────────────────────────────────┘
```

---

## Camada 1: Samples de áudio

### Opção A: Samples CC0 (Creative Commons Zero)

| Sample | Fonte | Tamanho | Qualidade |
|--------|-------|---------|-----------|
| Conversa | Freesound.org / LibriVox | ~30s WAV | Alta |
| Música | Free Music Archive | ~30s WAV | Alta |
| Ruído urbano | Freesound.org | ~30s WAV | Alta |

**Vantagem:** Zero custo, qualidade real, licença livre  
**Desvantagem:** Precisa baixar e hospedar os arquivos

### Opção B: TTS (Text-to-Speech)

Usar Web Speech API para gerar fala sintética mas realista:
```javascript
const utterance = new SpeechSynthesisUtterance("Hello, this is a test of hearing loss simulation.");
speechSynthesis.speak(utterance);
```

**Vantagem:** Sem arquivos externos, sempre disponível  
**Desvantagem:** Voz robótica, depende do browser

### Opção C: Música gerada com mais realismo

Em vez de sine waves simples, usar:
- Múltiplos osciladores com frequências aleatórias
- Ruído rosa (mais natural que branco)
- Envelopes ADSR (attack, decay, sustain, release)
- Modulação de amplitude para ritmo

### Recomendação

**Opção A (CC0) como padrão + Opção C como fallback.** Samples reais são sempre superiores a qualquer simulação.

---

## Camada 2: Motor de Simulação

### Opção A: 3D Tune-In Toolkit (preferencial)

**O que é:** Toolkit open-source (EU-funded, Imperial College London) para simulação de perda auditiva.

**Features:**
- Multi-band dynamic range compressor/expander
- Automatic configurator from audiogram
- Frequency smearing (Baer & Moore model)
- Temporal distortion (jitter)
- Binaural HRTF

**JavaScript wrapper:** https://github.com/3DTune-In/3dti_AudioToolkit_JavaScript

**Integração:**
```javascript
import AudioToolkit from '@reactify/3dti-toolkit';

const toolkit = AudioToolkit();
const hearingLossSimulator = new toolkit.CHearingLossSim();

// Configurar a partir do audiograma
hearingLossSimulator.ConfigureFromAudiogram(thresholds);

// Processar áudio
toolkit.HearingLossSim_Process(hearingLossSimulator, inputBuffers, outputBuffers);
```

**Vantagem:** Simulação realista, scientificamente fundamentada  
**Desvantagem:** GPLv3 (non-commercial free), integração complexa

### Opção B: Web Audio API nativo (fallback)

Melhorar a implementação atual:
- Usar `DynamicsCompressorNode` em vez de peaking filters
- Adicionar `ConvolverNode` para simular distorção
- Implementar noise floor (tinnitus simulado)
- Usar `StereoPannerNode` para assimetria

**Vantagem:** Sem dependências, funciona em qualquer browser  
**Desvantagem:** Menos realista que 3DTI

### Recomendação

**Opção A como padrão + Opção B como fallback.** Se 3DTI não carregar, usar Web Audio nativo melhorado.

---

## Camada 3: UI

### Problemas atuais e soluções

| Problema | Solução |
|----------|---------|
| toggleNormal() usa parsing de texto | ✅ Já corrigido (variável `currentAudioType`) |
| Footer "25 scripts" | ✅ Já corrigido (27) |
| iOS Safari AudioContext | Adicionar `onclick` handler que chama `audioCtx.resume()` |
| Share buttons podem ser bloqueados | Usar Web Share API como fallback |
| Não tem loading indicator | Adicionar spinner enquanto samples carregam |

---

## Implementação recomendada

### Fase 1 (agora — 1 hora)
1. ✅ Footer corrigido
2. ✅ toggleNormal corrigido
3. Adicionar samples CC0 de Freesound.org
4. Melhorar geração de áudio (noise + ADSR envelopes)

### Fase 2 (depois — 1 dia)
1. Integrar 3DTI Toolkit JS
2. Implementar fallback Web Audio nativo
3. Adicionar loading indicator

### Fase 3 (futuro — 1 semana)
1. Binaural rendering (HRTF)
2. Tinnitus simulation
3. User audiogram upload → personalized simulation

---

## Evitando erros da conversa passada

### Erro 1: Tudo num script grande
**Solução:** Separar em módulos: `audio-engine.js`, `samples.js`, `ui.js`

### Erro 2: Não testar antes de commitar
**Solução:** Testar cada camada isoladamente antes de integrar

### Erro 3: Find-and-replace em HTML
**Solução:** Usar templates (Jinja2) ou gerar HTML via Python

### Erro 4: Não verificar consistência
**Solução:** Checklist antes de commit:
- [ ] Footer atualizado?
- [ ] Variáveis globais corretas?
- [ ] Samples carregam?
- [ ] Simulação funciona?
- [ ] Mobile funciona?

---

## Checklist de credibilidade (para o paper/API)

| Item | Verificar |
|------|-----------|
| Disclaimer presente | "This simulation shows only frequency sensitivity loss" |
| Fonte dos dados | "NHANES, 26,583 audiograms" |
| Método | "HDBSCAN + row-centering" |
| Limitações | "Not a clinical tool" |
| Licença | MIT |
| Contato | Email real |
