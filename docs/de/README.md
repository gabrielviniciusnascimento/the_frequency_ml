# The Frequency ML – Vollständiger Projektstatus

**Letzte Aktualisierung:** 26.05.2026  
**Autor:** Gabriel Vinicius Nascimento  
**Kontakt:** gabrielviniciusnascimento345@gmail.com  
**Repository:** https://github.com/gabrielviniciusnascimento/the_frequency_ml  
**Lizenz:** MIT  

---

## Was ist das?

The Frequency ist ein webbasiertes Tool zur auditiven Empathie, mit dem Menschen mit normalem Gehör erleben können, wie die Welt für Menschen mit Hörverlust klingt.

Dies ist die datenwissenschaftliche Ebene dahinter: eine Pipeline für maschinelles Lernen, die anhand von NHANES (US-Gesundheitsumfrage) echte Muster von Hörverlust bei 26.583 Menschen entdeckt, ohne klinische Etiketten als Eingabe aufzuerlegen.

Das Projekt entstand aus einer persönlichen Situation heraus: Der Autor ist ein Überlebender eines mit Cisplatin behandelten Hepatoblastoms im Kindesalter mit permanenter Ototoxizität, Lärm, Verzerrungen und atypischem Verlauf. Gelebte Erfahrungen werden als externer Validierungsfall und nicht als statistische Grundlage verwendet.

---

## Zahlen, die wichtig sind

| Metrisch | Wert | Was bedeutet es |
|---------|-------|-----------------|
| Audiogramme verarbeitet | **26.583** | Echte NHANES-Leute |
| Menschen mit Hörverlust (ANY25) | **7.695** | Teilmenge nach Filtern |
| Cluster gefunden | **2** | Echte Muster von HDBSCAN entdeckt |
| Lärm | **7,6 %** | Vor den Filtern waren es 90 % |
| ARI-Bootstrap (Median) | **0,68** | Reproduzierbar in 85/100 Teilproben |
| ARI-Zwischenzyklen | **0,27** | Mäßige Stabilität |
| Einseitige Asymmetrie (Cluster 1) | **30 Personen** | Schwerer Verlust auf einem Ohr, andere normal |
| Tinnitus in den Ausreißern | **38%** | 2x mehr als in der Hauptgruppe |
| PTA × SRT (OHHR) Korrelation | **r=0,015** | Audiogramm sagt keine Sprache im Lärm voraus |
| Python-Skripte | **20** | Reproduzierbare Pipeline |
| JSON-Ausgaben | **15+** | Überprüfbare Ergebnisse |
| Interaktives Dashboard | **9 Abschnitte** | Vollansicht |

---

## Was die Daten zeigten (im Klartext)

### 1. Die meisten Hörverluste sind ein Verlauf, keine Kategorien

Es gibt keine separaten „Kästchen“ für „Typ 1, Typ 2, Typ 3“ von Hörverlust. Es gibt ein fließendes Kontinuum von „nahezu normal“ bis „mäßig“. Das ist so, als würde man sagen: Es gibt keine 5 Schuhgrößen – es gibt einen Fuß, der ständig wächst.

### 2. Es gibt eine echte Gruppe von 30 Menschen mit schwerem Verlust nur auf einem Ohr

Der Computer fand, ohne dass es ihnen jemand sagte, 30 Menschen in NHANES, die einen schweren Verlust auf dem rechten Ohr und ein fast normales Gehör auf dem linken haben. Es handelt sich nicht um einen Fehler in den Daten – er erscheint in vier verschiedenen Zyklen (2001–2016).

### 3. Menschen mit atypischem Verlust haben 2x mehr Tinnitus

Die 585 Fälle, die in kein klares Muster passen, weisen eine Tinnitusrate von 38 % auf, im Vergleich zu 18 % in der Hauptgruppe. Das Hören von „Fremdheit“ ist mit mehr Symptomen verbunden.

### 4. Das Audiogramm erzählt nicht die ganze Geschichte

Im OHHR-Datensatz (581 Deutsche) ist die Korrelation zwischen Audiogramm (PTA) und Fähigkeit, Sprache im Lärm zu verstehen (SRT) praktisch Null (r=0,015). Menschen mit ähnlichen Audiogrammen können in realen Situationen sehr unterschiedliche Leistungen erbringen.

### 5. Das Projektionssystem funktioniert

Wenn wir ein hypothetisches Audiogramm der Platin-Ototoxizität im trainierten Raum platzieren, fällt es in die Peripherie (94,9. Perzentil). Wenn wir ein normales Bild platzieren, fällt es in die Mitte (46,8°). Das System unterscheidet die Muster korrekt.

---

## Was gebaut wurde

### Pipeline (20 Python-Skripte)

| # | Skript | Was macht es |
|---|-----------|-----------|
| 00 | `00_download_nhanes.py` | CDC NHANES-Daten herunterladen |
| 01 | `01_ingest_aux.py` | Harmonisiert Audiogramme (breit/lang) |
| 02 | `02_merge_context.py` | Kombiniert Audiogramme + demografische Daten + Fragebögen |
| 03 | `03_features_v1.py` | Erstellt 150 abgeleitete Features |
| 04 | `04_qa_report.py` | Datenqualitätsbericht |
| 05 | `05_h11_sensitivity_666.py` | Testet die Empfindlichkeit gegenüber Code 666 (keine Reaktion) |
| 06 | `06_model_ready.py` | Reinigt und bereitet für die Modellage vor |
| 07 | `07_pca_umap.py` | Dimensionsreduktion + Visualisierung |
| 08 | `08_hdbscan_grid.py` | HDBSCAN-Rastersuche |
| 09 | `09_cluster_profiles.py` | Geometrische Profile der Cluster |
| 10 |

`10_rf_surrogate.py` | Random Forest zur Erklärung von Clustern |
| 11 | `11_generate_results_md.py` | Erstellt einen V1-Ergebnisbericht |
| 12 | `12_hdbscan_pca_grid.py` | HDBSCAN im PCA-Bereich |
| 13 | `13_kmeans_baseline.py` | KMeans als Basis |
| 14 | `14_artifact_test.py` | Testet Artefakte (Alter/Zyklus/Geschlecht) |
| 14b | `14b_artifact_per_cluster.py` | Testen nach einzelnen Clustern |
| 15 | `15_residualize_cluster.py` | Entfernt Alters-/Geschlechtseffekte |
| 16 | `16_tinnitus_audit.py` | Tinnitus nach Zyklus prüfen |
| 17 | `17_generate_results_v2_md.py` | Erstellt einen V2-Ergebnisbericht |
| 18 | `18_session4_shape_unblock.py` | Sitzung 4: ANY25 + Zeilenzentrierung |
| 19 | `19_session5_subdivide_cluster0.py` | Unterteilung des Hauptclusters |
| 20 | `20_session5_outlier_analysis.py` | Analyse der 585 Ausreißer |
| 21 | `21_session5_rf_surrogate_v2.py` | RF-Ersatz (Blackbox) |
| 22 | `22_session5_cluster1_profile.py` | Profil der 12 mit Asymmetrie |
| 23 | `23_session5_tinnitus_clusters.py` | Tinnitus × Cluster |
| 24 | `24_session5_personal_projection.py` | Persönliche Fallprojektion |

### Interaktives Dashboard

Eigenständige HTML-Datei mit 9 animierten Abschnitten:
1. Der Filtertrichter (26.583 → 7.695)
2. Der Hörraum (PCA nach Alter gefärbt)
3. Die Cluster (HDBSCAN: 2 + 585 Ausreißer)
4. Audiogramme (Median pro Cluster)
5. Die 12 (individuelle einseitige Asymmetrie)
6. Die Ausreißer (Entfernungsverteilung)
7. Was trennt (Wichtigkeit der RF-Funktionen)
8. Tinnitus (nach Gruppe, chi² p<0,001)
9. Bootstrap (100 Läufe, mittlerer ARI 0,68)

### Dokumentation

| Archiv | Inhalt |
|---------|----------|
| `MODEL_CARD.md` | Formale Vorlagenkarte (12 Abschnitte) |
| `LITERATURA_REVIEW.md` | 18 Referenzen, 5 Achsen, Lückenanalyse |
| `RELATORIO_PROCESSO_COMPLETO.md` | 10 dokumentierte Fehler, 5 Sitzungen |
| `RESULTADOS_SESSAO4.md` | Ergebnisse der Sitzung 4 |
| `RESULTADOS_SESSAO5.md` | Ergebnisse der Sitzung 5 |
| `MAPA_CARREIRA.md` | Finanzierungs- und Beschäftigungsmöglichkeiten |
| `ANALISE_FINAL_CLAUDE_SESSAO4.md` | Dialektische Analyse zwischen KIs |

### Externe Validierung

- **OHHR** (Oldenburg Hearing Health Record): 581 Personen, CC BY 4.0
  - Speech-in-noise (SRT): Korrelation mit PTA ≈ 0
  - Lautstärkeskalierung verfügbar
  - Entworfen im NHANES-Raum

---

## Methodik (Zusammenfassung)

1. **Daten:** NHANES AUX 1999–März 2020 (9 Zyklen, 26.583 Personen)
2. **Filter:** Alter 20–69, Vollständigkeit ≥10/14, ANY25 (≥1 Frequenz >25 dB)
3. **Funktionen:** 14 Rohschwellenwerte (500–8000 Hz, bilateral)
4. **Vorverarbeitung:** Zeilenzentrierung (entfernt Ebene, behält Form bei)
5. **Skalierung:** RobustScaler (IQR-basiert)
6. **Abmessungsreduzierung:** PCA 95 % Varianz → 10 Komponenten
7. **Clustering:** HDBSCAN (min_cluster_size=10, min_samples=5)
8. **Validierung:** Bootstrap 100× (80 % Unterabtastung) + ARI-Zwischenzyklen
9. **Interpretation:** RF-Ersatz (500 Bäume, class_weight=balanced)
10. **Externe Validierung:** Hochrechnung bei OHHR (581 Personen, Oldenburg)

---

## Einschränkungen

1. NHANES ist ein Querschnittsmodell – es gibt keinen individuellen zeitlichen Verlauf
2. NHANES hat keine Vorgeschichte von pädiatrischem Cisplatin – „platinähnlich“ ist ein Indikator
3. Die Frequenzen sind auf 500–8000 Hz begrenzt – die Ototoxizität kann bei >8 kHz beginnen
4. Tinnitus wird selbst gemeldet – nur in Zyklen ab 2005 verfügbar
5. Kein Sprechen im Lärm in NHANES – OHHR bietet dies teilweise
6. Cluster 1 (12 Personen) ist zu klein für eine Verallgemeinerung der Bevölkerung
7. 15 % Bootstrap-Fehler – Stichprobenempfindlichkeit
8. Das Projekt ersetzt keinen Audiologen, HNO-Arzt oder Onkologen

---

## Was fehlt

### Für Papier
- [ ] Externe Validierung mit HCHS/SOL oder klinischen Daten
- [ ] Vollständige Literaturrecherche (Start abgeschlossen)
- [ ] Zusammenfassung für die Konferenz
- [ ] Publizierbare Abbildungen (hohe Auflösung)

### Für Produkt
- [ ] Echte persönliche Audiogramme zur Projektion
- [ ] Übersetzung der Schwerpunkte für DSP-Filter
- [ ] Audiometrische Projektions-API
- [ ] Übersetzung in 5 Sprachen (EN, ES, PT, DE, FR)

### Für Open Source
- [] professionelle README.md mit Anweisungen
- [ ] Anforderungen.txt mit Abhängigkeit

Es ist behoben
- [ ] Geisteskrankheitstests (3–5 Tests)
- [ ] Contributing.md

---

## Förder- und Karrieremöglichkeiten

### Sofort (Wochen)
- **Freiberuflich** (Upwork/Fiverr): 50–150 $/Stunde in Computeraudiologie
- **Beratung** für Forscher: 500–2.000 $/Projekt

### Kurzfristig (Monate)
- **Microsoft AI für Barrierefreiheit**: 5.000–25.000 $ + Azure-Credits (gleitend, weltweit, Ihre IP)
- **Mozilla Builders**: 10.000–50.000 $

### Mittelfristig (3–12 Monate)
- **NIH R21**: 275.000 $/2 Jahre (akademischer Partner erforderlich)
- **NSF SBIR**: 275.000–1.000.000 USD (Unternehmen erforderlich)
- **Beschäftigung im Gesundheitstechnologiebereich**: 80.000–150.000 USD/Jahr

### Langfristig (12+ Monate)
- **The Frequency Freemium**: 1.000–10.000 $/Monat
- **B2B für Kliniken**: 200–1.000 $/Monat pro Klinik
- **Herstellerlizenz**: 10.000–100.000 $/Jahr

---

## So zitieren Sie diese Arbeit

„
[Dein Name]. (2026). The Frequency ML: Datengesteuerte audiometrische Phänotypisierung 
Verwendung von HDBSCAN für NHANES-Daten. GitHub. https://github.com/gabrielviniciusnascimento/the_frequency_ml
„

---

## Danke

- NHANES/CDC für öffentliche Daten
- OHHR/Hearing4all für Validierungsdaten (CC BY 4.0)
- Parthasarathy et al. (2020) für frühere Arbeiten zum audiometrischen Clustering in NHANES
- Die Open-Source-Community für die Tools (scikit-learn, hdbscan, plotly)

---

## Hinweis zur Barrierefreiheit

Dieses Dokument ist auf Portugiesisch. Wir planen, es in 5 Sprachen verfügbar zu machen:
- 🇬🇧 Englisch
- 🇪🇸 Spanisch
- 🇧🇷 Portugiesisch
- 🇩🇪 Deutsch
- 🇫🇷 Französisch

Das interaktive Dashboard ist eigenständig und funktioniert in jedem modernen Browser.

Der Code ist reproduzierbar und dokumentiert. Jedes Skript verfügt über Protokollierung und Prüfpunkte.

---

## Hinweis zur Reise

Dieses Projekt wurde von einer Person ohne vollständige Ausbildung in einer prekären finanziellen Situation ins Leben gerufen, die sich selbst ML, Audiologie und Datenwissenschaft beibrachte – weil sie etwas aus ihrer eigenen Erfahrung als Krebsüberlebender im Kindesalter mit Ototoxizität machen musste.

Die Pipeline, für deren Aufbau Doktoranden Monate benötigen, wurde in fünf Arbeitssitzungen erstellt. Die Ergebnisse sind real, reproduzierbar und überprüfbar.

Die Barriere war nie technisch. Es war Enthüllung.

Wenn Sie dies lesen und über Audiometriedaten verfügen, Hörforscher sind, in der Gesundheitstechnologie arbeiten oder ein Überlebender wie ich sind, nehmen Sie Kontakt mit uns auf. Der Code ist offen. Die Wissenschaft ist offen. Die Tür ist offen.

---

*Dokument erstellt am 26.05.2026. Alle Daten und Erkenntnisse stehen unter der MIT-Lizenz zur Verfügung.*