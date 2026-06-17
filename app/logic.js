/* Lógica pura da ferramenta — sem DOM, sem Web Audio. Isolada aqui para ser
   testável (tests/app_logic.test.js) e reusada pelo index.html (window.TF).
   Os limiares dos PROFILES são derivados do NHANES (ver app/audio/CREDITS.md e o preprint). */
(function (root) {
  "use strict";

  const FREQS = [500, 1000, 2000, 3000, 4000, 6000, 8000];
  const FLABELS = ["500", "1k", "2k", "3k", "4k", "6k", "8k"];
  const PROFILES = {
    0: { name: "Perda gradual nos agudos",
         R: [14.5, 14.5, 16.5, 24.1, 28.1, 31.2, 33.1], L: [14.6, 14.6, 16.6, 25.6, 30.6, 32.1, 33.6] },
    1: { name: "Perda severa em uma orelha",
         R: [68.6, 70.0, 72.5, 77.5, 85.0, 85.0, 87.5], L: [10.0, 7.5, 12.5, 15.0, 22.5, 25.0, 32.5] },
    2: { name: "Queda acentuada nos agudos",
         R: [26.9, 26.9, 28.9, 43.2, 53.2, 56.0, 53.2], L: [27.9, 27.9, 29.9, 46.0, 56.0, 58.7, 56.0] },
  };
  const Q = 1.4, MASTER_GAIN = 0.3, VOL_MAX_GAIN = 4.0, DISCLAIMER_PCT = 80;

  // Ganho do ruído relativo à fala para um dado SNR (dB). +SNR = ruído mais baixo.
  function snrToGain(db) { return Math.pow(10, -db / 20); }

  // Posição do slider de volume (0–100%) → ganho linear do master (limiter protege o pico).
  function volGain(pct) { return (pct / 100) * VOL_MAX_GAIN; }

  // Maior perda (dB) entre as duas orelhas de um perfil — ancora o disclaimer.
  function maxLossDb(p) { return Math.max.apply(null, p.R.concat(p.L)); }

  // O disclaimer de volume só aparece com perfil selecionado, em modo COM PERDA e volume alto.
  function showDisclaimer(pct, hasProfile, normal) {
    return pct >= DISCLAIMER_PCT && !!hasProfile && !normal;
  }

  // Categoria da dica de fala-no-ruído, ancorada no limiar normal (~+2 dB) do QuickSIN.
  function sinHintKey(snr) {
    if (snr >= 12) return "easy";
    if (snr >= 5) return "comfort";
    if (snr >= 0) return "edge";
    if (snr >= -6) return "below";
    return "noise";
  }

  // Decisão real-vs-sintético do bufFor: usa gravação real quando ela já carregou.
  function useReal(kind, has) {
    has = has || {};
    if ((kind === "voice" || kind === "voice_l") && has.voice) return true;
    if (kind === "babble" && has.babble) return true;
    if (kind === "street" && has.street) return true;
    if (kind === "music" && has.music) return true;
    return false;
  }

  const TF = { FREQS, FLABELS, PROFILES, Q, MASTER_GAIN, VOL_MAX_GAIN, DISCLAIMER_PCT,
               snrToGain, volGain, maxLossDb, showDisclaimer, sinHintKey, useReal };

  if (typeof module !== "undefined" && module.exports) module.exports = TF;
  else root.TF = TF;
})(typeof self !== "undefined" ? self : globalThis);
