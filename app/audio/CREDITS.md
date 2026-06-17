# Audio credits & licences

The two audio clips bundled here are short excerpts of **public, openly-licensed**
datasets, used as the speech and the background-noise masker in the speech-in-noise
demo. Both are redistributable; the exact terms, sources and the precise excerpt used
are recorded below so the choice can be checked.

Retrieved: **2026-06-16**.

---

## `speech_harvard_osr.wav` — the speech

- **Source:** Open Speech Repository — American English, file `OSR_us_000_0010_8k.wav`.
  <https://www.voiptroubleshooter.com/open_speech/american.html>
- **Content:** Harvard / IEEE phonetically-balanced sentences (IEEE, *Recommended
  Practice for Speech Quality Measurements*, 1969 — the standard speech-test material).
  The file holds **Harvard List 1** (confirmed by ear against the canonical list — the
  OSR page itself labels it only "Harvard sentences"). The clip is the three opening
  sentences: *"The birch canoe slid on the smooth planks. Glue the sheet to the dark
  blue background. It's easy to tell the depth of a well."*
- **Excerpt:** seconds **0.0–9.7** of the source file (the file starts straight into
  sentence 1 — there is no spoken announcement; cut ends in the pause after sentence 3),
  mono, kept at the original **8 kHz** sample rate (telephony band — the band these
  sentences are designed for). Trimmed with the Python standard library only
  (`py trim_audio.py OSR_us_000_0010_8k.wav speech_harvard_osr.wav 0.0 9.7`) — a plain
  RIFF/`fmt`/`data` WAV. (An ffmpeg cut was tried but it injected a `LIST/INFO` metadata
  chunk that the browser's `decodeAudioData` rejected, so the stdlib path is used.)
- **Usage terms (verbatim):** *"The material on this site is freely available for use
  in VoIP testing, research, development, marketing and any other reasonable
  application."*

> Note: 8 kHz means there is no energy above 4 kHz, so the 6 k/8 k audiogram bands act
> mostly on the noise, not the speech. This is honest and intentional — flagged here so
> nobody mistakes the demo for full-band audio.

## `babble_demand_presto.wav` — the background noise

- **Source:** DEMAND — *Diverse Environments Multi-channel Acoustic Noise Database*,
  environment **PRESTO** (a public restaurant), channel `ch01`, 16 kHz version.
  Zenodo record 1227121 — <https://zenodo.org/records/1227121>
  (downloaded zip MD5 `b98d2e6854eeebb397f29a8ad7457092`, matches the published MD5).
- **Excerpt:** seconds **30.0–38.0** of `ch01`, mono, original **16 kHz** sample rate.
- **Licence:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Citation:** J. Thiemann, N. Ito, E. Vincent, *"The Diverse Environments
  Multi-channel Acoustic Noise Database (DEMAND): A database of multichannel
  environmental noise recordings,"* Proc. Meetings on Acoustics (ICA 2013), vol. 19,
  035081, 2013.

## `street_demand_straffic.wav` — the street ("Rua")

- **Source:** DEMAND, environment **STRAFFIC** (street with traffic), channel `ch01`,
  16 kHz version. Zenodo record 1227121 — <https://zenodo.org/records/1227121>
  (downloaded zip MD5 `2efa87262f272bbf9ba578088e81939c`).
- **Excerpt:** seconds **30.0–38.0** of `ch01`, mono, original **16 kHz** sample rate.
- **Licence:** **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
- **Citation:** same as PRESTO above (Thiemann, Ito & Vincent, 2013).

## `music_sousa_usmb.ogg` — the music

- **Source:** *"The Stars and Stripes Forever"* (J. P. Sousa, 1896), performed by the
  **United States Marine Band ("The President's Own")**, 2017, cond. Lt. Col. Jason K.
  Fettig. Via Wikimedia Commons.
  <https://commons.wikimedia.org/wiki/File:Sousa%27s_%22The_Stars_and_Stripes_Forever%22_-_United_States_Marine_Band_(2017).ogg>
  Official source: <https://www.marineband.marines.mil/Audio-Resources/>
- **Excerpt:** seconds **186–195** of the recording (the final strain — piccolo
  obbligato + cymbals, chosen for its high-frequency content so treble loss is audible),
  downmixed to mono, **48 kHz** full-band, short fades, re-encoded to Ogg Vorbis with
  ffmpeg (`ffmpeg -ss 186 -t 9 -i src.ogg -ac 1 -ar 48000 -af afade... -c:a libvorbis -q:a 4`).
- **Licence:** **Public domain.** The 1896 composition is PD (author died 1932); the
  recording is a work of the U.S. federal government (U.S. Marine Band) and therefore PD.

---

## Scientific reference for the SNR scale

The speech-in-noise scale (normal listeners reach ~50% words at **+2 dB SNR** in
four-talker babble; "SNR loss" of 3–7 / 7–15 / >15 dB = mild / moderate / severe) is
from the **QuickSIN** test: M. C. Killion, P. A. Niquette, G. I. Gudmundsen,
L. J. Revit, S. Banerjee, *"Development of a quick speech-in-noise test for measuring
signal-to-noise ratio loss in normal-hearing and hearing-impaired listeners,"*
*J. Acoust. Soc. Am.* 116(4), 2395–2405, 2004.

## Audiogram data

The example thresholds in the tool are derived from **NHANES** audiometry (US public
domain) — see the repository's methods audit and preprint.
