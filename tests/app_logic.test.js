/* Testes unitários da lógica de UI/UX da ferramenta de empatia.
   Rodar: node --test tests/app_logic.test.js  (a partir da raiz do repo) */
"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const TF = require("../app/logic.js");

test("snrToGain: +SNR abaixa o ruído, 0 dB = paridade, -SNR levanta", () => {
  assert.equal(TF.snrToGain(0), 1);
  assert.ok(TF.snrToGain(6) < 1 && Math.abs(TF.snrToGain(6) - 0.5012) < 1e-3); // ~ -6 dB
  assert.ok(TF.snrToGain(-6) > 1 && Math.abs(TF.snrToGain(-6) - 1.995) < 1e-3);
});

test("volGain: 0% = mudo, 70% padrão, 100% = teto", () => {
  assert.equal(TF.volGain(0), 0);
  assert.equal(TF.volGain(100), TF.VOL_MAX_GAIN);
  assert.ok(TF.volGain(70) > TF.volGain(60)); // monotônico
  assert.equal(TF.volGain(70), 2.8);
});

test("showDisclaimer: só com perfil, em perda e volume >= limiar", () => {
  assert.equal(TF.showDisclaimer(80, true, false), true);
  assert.equal(TF.showDisclaimer(79, true, false), false);   // abaixo do limiar
  assert.equal(TF.showDisclaimer(95, true, true), false);    // modo NORMAL não mostra
  assert.equal(TF.showDisclaimer(95, false, false), false);  // sem perfil não mostra
});

test("maxLossDb: pega a pior banda entre as duas orelhas", () => {
  assert.equal(TF.maxLossDb(TF.PROFILES[1]), 87.5); // orelha direita severa
  assert.equal(TF.maxLossDb(TF.PROFILES[0]), 33.6);
});

test("sinHintKey: fronteiras ancoradas no ~+2 dB do QuickSIN", () => {
  assert.equal(TF.sinHintKey(12), "easy");
  assert.equal(TF.sinHintKey(5), "comfort");
  assert.equal(TF.sinHintKey(0), "edge");
  assert.equal(TF.sinHintKey(-3), "below");
  assert.equal(TF.sinHintKey(-7), "noise");
});

test("useReal: usa gravação real só quando carregou; fallback sintético", () => {
  assert.equal(TF.useReal("voice", { voice: true }), true);
  assert.equal(TF.useReal("voice_l", { voice: true }), true); // fala da SIN reusa a real
  assert.equal(TF.useReal("babble", { babble: true }), true);
  assert.equal(TF.useReal("street", { street: true }), true);
  assert.equal(TF.useReal("music", { music: true }), true);
  assert.equal(TF.useReal("voice", {}), false);              // ainda não carregou
  assert.equal(TF.useReal("music", { voice: true }), false); // não confunde fontes
});

test("PROFILES: estrutura válida (7 bandas por orelha, alinhadas às FREQS)", () => {
  assert.equal(TF.FREQS.length, 7);
  for (const k of Object.keys(TF.PROFILES)) {
    const p = TF.PROFILES[k];
    assert.equal(p.R.length, 7, `R do perfil ${k}`);
    assert.equal(p.L.length, 7, `L do perfil ${k}`);
    assert.equal(typeof p.name, "string");
    p.R.concat(p.L).forEach((v) => assert.ok(v >= 0 && v <= 120, "limiar em faixa plausível"));
  }
});
