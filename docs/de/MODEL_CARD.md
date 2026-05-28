# Modellkarte – Die Frequenz ML

**Version:** 1.0  
**Datum:** 26.05.2026  
**Autoren:** Das Frequency-Team  
**Status:** Experimentell – keine klinische Kennzeichnung  

---

## 1. Übersicht

| Feld | Wert |
|-------|-------|
| **Modellname** | Audiometrisches HDBSCAN-Clustering (nur Form) |
| **Typ** | Unüberwachtes Clustering |
| **Aufgabe** | Entdecken Sie latente Muster von Hörverlust in Bevölkerungsdaten |
| **Trainingsdaten** | NHANES AUX 1999–März 2020 (9 Zyklen) |
| **N-Training** | 7.695 (nach Filtern) |
| **Funktionen** | 14 Rohschwellen (500–8000 Hz, bilateral) |
| **Metriken** | ARI, Ausreißeranteil, Gini-Wichtigkeit |
| **Verwendungszweck** | Auditive Empathieforschung + Simulation (The Frequency) |
| **Verwendung nicht empfohlen** | Individuelle klinische Diagnose |

---

## 2. Daten

### 2.1 Quelle

NHANES (National Health and Nutrition Examination Survey), CDC/NCHS. US-amerikanische Querschnittsbevölkerungsumfrage mit Reintonaudiometrie pro Ohr/Frequenz.

| Zyklus | Archiv | N brutto | Frequenzen |
|-------|--------|---------|-------------|
| 1999–2000 | AUX1.xpt | 1.807 | 500–8000 Hz |
| 2001–2002 | AUX_B.xpt | 2.046 | 500–8000 Hz |
| 2003–2004 | AUX_C.xpt | 1.889 | 500–8000 Hz |
| 2005–2006 | AUX_D.xpt | 3.034 | 500–8000 Hz |
| 2007–2008 | AUX_E.xpt | 1.210 | 500–8000 Hz |
| 2009–2010 | AUX_F.xpt | 2.368 | 500–8000 Hz |
| 2011–2012 | AUX_G.xpt | 4.500 | 500–8000 Hz |
| 2015–2016 | AUX_I.xpt | 4.582 | 500–8000 Hz |
| 2017–März 2020 | P_AUX.xpt | 5.147 | 500–8000 Hz |
| **Gesamt** | | **26.583** | |

### 2.2 Angewandte Filter

| Filter | Begründung | Vorher | Nach |
|--------|------------|-------|--------|
| Alter 20–69 | Zyklen mit unterschiedlicher Berechtigung entfernen (Jugendliche, 70+) | 26.583 | 14.824 |
| Vollständigkeit ≥10/14 | Sorgen Sie für ausreichend Daten pro Person | 14.824 | 13.433 |
| ANY25 (≥1 Frequenz >25 dB) | Entfernen Sie die gesunde „Sonne“, die die Dichte verschluckt hat | 13.433 | 7.695 |

### 2.3 Bekannte Störvariablen

| Variable | Auswirkungen | Behandlung |
|----------|---------|------------|
| Alter | Stark (R² ~0,57 in PTA_high) | Filter 20–69 + Zeilenzentrierung |
| Zyklus | Mäßig (Cramérs V ~0,16) | Validierung pro Zyklus (ARI) |
| Sex | Schwach (Cramérs V ~0,12) | Unkontrolliert (Zukunft) |
| 666 (keine Antwort) | 511 Zeilen (1,9 %) | Primäre Richtlinie: NaN + Flag |

---

## 3. Vorverarbeitung

### 3.1 Umgang mit Schwellenwerten

| Code | Bedeutung | Behandlung |
|--------|-------------|------------|
| -10 bis 120 dB | Gültig | Konserviert |
| 666 | Keine Reaktion (starke Zensur) | → NaN + Flag |
| 888 | | konnte nicht abgerufen werden → NaN |
| Andere | Fehlt | → NaN |

### 3.2 Zeilenzentrierung

Für jedes einzelne *i*:

$$\mu_i = \frac{1}{14} \sum_{f \in F} T_{i,f}$$

$$T^{shape}_{i,f} = T_{i,f} - \mu_i$$

Es entfernt das durchschnittliche „Ausmaß“ des Verlusts (wie viel die Person im Durchschnitt verliert) und behält die „Form“ der Kurve bei (wo der Verlust am größten/kleinsten ist).

### 3.3 Skalierung

RobustScaler (IQR-basiert, quantile_range=(25, 75)). Es setzt keine Normalität voraus. Ausreißerresistent.

### 3.4 Dimensionsreduzierung

PCA mit 95 % Varianz erklärt. Ergebnis: 10 Komponenten (von 14).

---

## 4. Modell

### 4.1 Algorithmus

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).

### 4.2 Hyperparameter

| Parameter | Wert | Begründung |
|-----------|-------|---------------|
| min_cluster_size | 10 | Kleinster Wert, der Struktur findet (getestetes Raster: 5–200) |
| min_samples | 5 | HDBSCAN-Standard, getestet mit 3–20 |
| Metriken | euklidisch | Standard für kontinuierliche Daten |
| Clusterauswahlmethode | eom | Massenüberschuss (Standard) |
| core_dist_n_jobs | -1 | Parallelität |

### 4.3 Grid getestet

| min_cluster_size | min_samples | n_cluster | Noise_Fraktion |
|-----------------|-------------|------------|----------------|
| 5 | 3 | 12 | 0,048 |
| 5 | 5 | 4 | 0,088 |
| **10** | **5** | **2** | **0,076** |
| 15 | 5 | 2 | 0,083 |
| 20 | 5 | 0 | 1.000 |
| 30+ | irgendein | 0 | 1.000 |

---

## 5. Ergebnisse

### 5.1 Cluster gefunden

| Cluster | n | %

| Geometrische Beschreibung |
|---------|---|---|---------------------|
| 0 | 7.098 | 92,2 % | Leichter bis mäßiger Neigungsverlust, relativ symmetrisch beidseitig |
| 1 | 12 | 0,2 % | Schwere einseitige Asymmetrie (rechtes Ohr ~80 dB, linkes Ohr ~16 dB) |
| Lärm | 585 | 7,6 % | Heterogene Muster, mäßiger bis schwerer Verlust |

### 5.2 Metriken

| Metrisch | Wert | Interpretation |
|---------|-------|---------------|
| HDBSCAN-Rauschen | 7,6 % | Niedrig (war ~90 % vor den Filtern) |
| Zyklusübergreifender ARI | 0,27 | Konsistenz zwischen NHANES-Zyklen mit unterschiedlicher Eignung (misst die Robustheit gegenüber Protokoll-/Kohortenvariationen) |
| Bootstrap ARI (mittel) | 0,68 | Reproduzierbarkeit innerhalb von Teilstichproben derselben Population (misst die interne Stabilität) |
| Bootstrap ARI (bedingt) | 0,60 | Wenn Cluster auftreten (85 % der Teilstichproben) |

> **Hinweis:** Cross-Cycle-ARI und Bootstrap-ARI sind unterschiedliche Metriken. Die erste misst die Konsistenz zwischen verschiedenen Populationen (NHANES-Zyklen); der zweite misst die Reproduzierbarkeit innerhalb derselben Population. Beide werden aus Gründen der Transparenz gemeldet.
| RF AUC (Cluster 0 vs. 1) | 1,0 | Perfekte Trennung (unausgeglichene Klassen) |

### 5.3 Blackbox (RF-Ersatz)

Die 7 wichtigsten Unterscheidungsmerkmale (alle vom rechten Ohr):

| Funktion | Gini-Bedeutung |
|---------|----------------|
| thr_R_1000 | 0,2248 |
| thr_R_500 | 0,2203 |
| thr_R_2000 | 0,1453 |
| thr_R_4000 | 0,1175 |
| thr_R_3000 | 0,1174 |
| thr_R_6000 | 0,0711 |
| thr_R_8000 | 0,0427 |

### 5.4 Tinnitus

| Gruppe | Bewerten | n gültig |
|-------|------|----------|
| Cluster 0 | 18,3 % | 4.397 |
| Cluster 1 | 50,0 % | 8 |

> **Hinweis:** Die Tinnitusrate von Cluster 1 basiert auf N=8 Personen mit verfügbaren Daten. Als richtungsweisend und nicht statistisch eindeutig interpretieren.
| Ausreißer | 38,0 % | 308 |

Chi² p<0,001, Cramér's V=0,126.

---

## 6. Sensitivitätsanalyse (H11)

| Politik | Behandlung 666 | ARI vs. Nan | Auswirkungen |
|----------|----------------|---------------|---------|
| nan (primär) | 666 → NaN + Flag | — | Referenz |
| cap125 (alternativ) | 666 → 125 dB + Flag | 0,9914 | Minimum |

511 Linien betroffen (1,9 %). ARI 0,99 über Richtlinien hinweg → Ergebnis unempfindlich gegenüber 666-Behandlung.

### 6.2 ANY25 Filterempfindlichkeit

| Konfiguration | N | Cluster | Lärm | ARI vs. Grundschule |
|---------------|---|----------|-------|----------------|
| Mit ANY25 (primär) | 7.695 | 2 | 7,6 % | — |
| Ohne ANY25 | 13.433 | 2 | 4,4 % | 0,85 |

Der ANY25-Filter entfernt den „gesunden Kern“, verzerrt aber nicht die entdeckte Struktur. 98,9 % der Mitglieder von Cluster 0 und 75 % der Mitglieder von Cluster 1 bleiben über alle Filtereinstellungen hinweg erhalten.

### 6.3 OHHR-Pipeline-Konsistenz

| OHHRKonfiguration | N | Lärm | PTA×SRT r |
|------|---|-------|-----------|
| Ohne ANY25 | 581 | 53,0 % | 0,015 |
| Mit ANY25 | 537 | 54,0 % | 0,018 |

Der auf OHHR angewendete ANY25-Filter liefert praktisch identische Ergebnisse, was bestätigt, dass die Pipeline dieser Wahl standhält.

### 6.4 Bootstrap-Stabilität im 4D-Raum

| Raum | N dimmt | Mittlerer ARI | Läuft mit Clustern | SD |
|-------|--------|------------|-------------------|-----|
| 14D (vollständige Schwellenwerte) | 10 PCA | 0,68 | 85 % | ~0,40 |
| 4D (binauraler Mittelwert 500/1k/2k/4k) | 4 PCA | **0,74** | **100%** | **0,016** |

Der binaurale Mittelwertraum mit 4 Frequenzen ist *stabiler* als der vollständige 14-Frequenzraum. Dies ist der Raum, der für die externe OHHR-Validierung genutzt wird, wodurch der bevölkerungsübergreifende Vergleich gestärkt wird.



---

## 7. Validierung

### 7.1 Validierung pro Zyklus ( approximate_predict )

| Zyklus | n-Test | ARI |
|-------|------------|-----|
| 1999–2000 | 949 | 0,17 |
| 2001–2002 | 1.031 | 0,21 |
| 2003–2004 | 1.000 | 0,18 |
| 2011–2012 | 2.238 | 0,37 |
| 2015–2016 | 2.477 | 0,41 |

Durchschnittlicher ARI: 0,27. Neuere Zyklen (höheres N) weisen einen höheren ARI auf.

### 7.2 Bootstrap (100 Läufe × 80 %)

- 85 % der Teilproben fanden 2 Cluster
- Medianer ARI: 0,68
- Bedingter ARI (bei 2 Clustern): 0,60
- 15 % Fehler: Cluster 1 (12 Personen) bildet sich bei der Unterabtastung nicht

### 7,3 Va

Externer Handel – OHHR

**Ausgeführt.** OHHR (Oldenburg Hearing Health Record; Jafri et al., 2025): 581 Erwachsene (Durchschnittsalter 71, mittlerer PTA 45 dB), CC BY 4,0.

**Angewandte Pipeline:**
1. Extraktion der 4 gängigen Frequenzen mit NHANES (500, 1000, 2000, 4000 Hz)
2. Zeilenzentrierung (gleicher Vorgang wie NHANES)
3. RobustScaler mit NHANES-Parametern (nicht neu abgestimmt)
4. Projektion auf NHANES-trainiertes PCA (10 Komponenten)
5. „ approximate_predict “ mit NHANES HDBSCAN-Cluster

**Ergebnisse:**
- 53 % der OHHR fielen als Lärm auf (gegenüber 37,6 % bei NHANES) – erwartet, da OHHR älter und klinisch ist
- PTA × SRT-Korrelation: Pearson r=0,015, Spearman r=−0,007 (N=581)
- Interpretation: Audiogramm sagt keine Sprache im Lärm voraus („Faktor D“)

**Einschränkung:** OHHR trennt R/L nicht, was einen Asymmetrievergleich unmöglich macht. Die Frequenzen sind auf 500–4000 Hz begrenzt – aber das Bootstrapping in 4D zeigte eine *größere* Stabilität als 14D (ARI 0,74 vs. 0,68).

**Pipeline-Konsistenz:** OHHR mit ANY25-Filter (N=537): 54,0 % Rauschen, PTA×SRT r=0,018 – praktisch identisch mit ohne Filter (53,0 %, r=0,015).

---

## 8. Einschränkungen

### 8.1 Datenbeschränkungen

1. NHANES ist ein Querschnittsmodell – es gibt keinen individuellen zeitlichen Verlauf.
2. NHANES hat keine Vorgeschichte von pädiatrischem Cisplatin – „platinähnlich“ ist ein Ersatz, keine Bestätigung.
3. Die Frequenzen sind auf 500–8000 Hz begrenzt – die Ototoxizität kann bei >8 kHz beginnen.
4. Tinnitus wird selbst gemeldet (AUQ191) – nur verfügbar in Zyklen ab 2005.
5. Kein Sprechen im Lärm – NHANES misst nicht die funktionale Wahrnehmung.

### 8.2 Modellbeschränkungen

1. HDBSCAN reagiert empfindlich auf min_cluster_size – kleine Cluster bilden sich möglicherweise nicht.
2. Durch die Zeilenzentrierung wird das Niveau entfernt – es wird nicht erfasst, „wie schlimm“ der Verlust ist, sondern nur die Form.
3. 14 Dimensionen sind wenige – aber sie erfassen 95 % der Varianz.
4. Cluster 1 (12 Personen) ist zu klein für eine Verallgemeinerung der Bevölkerung.
5. Der Bootstrap-Fehler von 15 % zeigt die Stichprobenempfindlichkeit.

### 8.3 Ethische Einschränkungen

1. Kein Cluster erhielt eine klinische Kennzeichnung – es liegt an der Geometrie, nicht an der Diagnose.
2. Der persönliche Fall des Gründers wurde in der Schulung nicht verwendet.
3. Prävalenzen sollten nicht ohne Umfragegewichte abgeleitet werden.
4. Das Modell sollte nicht für individuelle klinische Entscheidungen verwendet werden.

---

## 9. Empfohlene Verwendung

| ✅ Kann | ❌ Darf nicht |
|---------|------------|
| Erforschung audiometrischer Standards | Individuelle klinische Diagnose |
| Auditive Empathie-Simulation | Prävalenzinferenz ohne Gewichte |
| Generierung klinischer Hypothesen | Audiologen ersetzen |
| Persönliche Fallvalidierung als externer Punkt | Persönliche Daten als statistische Grundlage nutzen |

---

## 10. Reproduzierbarkeit

### 10.1 Umgebung

„
Python 3.13+
numpy, pandas, scipy, scikit-learn, hdbscan, joblib, plotly
„

### 10.2 Skripte

20 Python-Skripte, fortlaufend nummeriert. Jedes Skript verfügt über Prüfpunkte (wird nicht erneut ausgeführt, wenn eine Ausgabe vorhanden ist).

### 10.3 Daten

Öffentliches NHANES XPT über CDC. URLs dokumentiert in „scripts/00_download_nhanes.py“.

### 10.4 Ausgänge

Über 15 JSON-Dateien mit vollständigen Ergebnissen. Alles aus den Skripten reproduzierbar.

---

## 11. Referenzen

- NHANES: https://wwwn.cdc.gov/nchs/nhanes/
- HDBSCAN: McInnes, L., Healy, J. (2017). Beschleunigtes hierarchisches dichtebasiertes Clustering. ICDM 2017.
- ARI: Hubert, L., Arabie, P. (1985). Partitionen vergleichen. Journal of Classification, 2(1), 193-218.

---

## 12. Kontakt

Die Frequenz – gabrielviniciusnascimento345@gmail.com

---

*Modellkarte erstellt am 26.05.2026. Im Training wurden keine klinischen Etiketten verwendet.*