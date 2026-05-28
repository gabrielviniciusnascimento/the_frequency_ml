# Carte modèle — La fréquence ML

**Version :** 1.0  
**Date :** 2026-05-26  
**Auteurs :** L'équipe Fréquence  
**Statut :** Expérimental — pas d'étiquette clinique  

---

## 1. Aperçu

| Champ | Valeur |
|-------|-------|
| **Nom du modèle** | Clustering audiométrique HDBSCAN (forme uniquement) |
| **Tapez** | Clustering non supervisé |
| **Tâche** | Découvrez les schémas latents de perte auditive dans les données démographiques |
| **Données de formation** | NHANES AUX 1999 – mars 2020 (9 cycles) |
| **Formation N** | 7 695 (après filtres) |
| **Caractéristiques** | 14 seuils bruts (500-8 000 Hz, bilatéraux) |
| **Mesures** | ARI, fraction aberrante, importance de Gini |
| **Utilisation prévue** | Recherche sur l'empathie auditive + simulation (The Frequency) |
| **Utilisation déconseillée** | Diagnostic clinique individuel |

---

## 2. Données

### 2.1 Origine

NHANES (Enquête nationale sur les examens de santé et de nutrition), CDC/NCHS. Enquête démographique transversale aux États-Unis, avec audiométrie tonale pure par oreille/fréquence.

| Cycles | Archives | N brut | Fréquences |
|-------|--------|---------|-------------|
| 1999-2000 | AUX1.xpt | 1 807 | 500-8 000 Hz |
| 2001-2002 | AUX_B.xpt | 2 046 | 500-8 000 Hz |
| 2003-2004 | AUX_C.xpt | 1 889 | 500-8 000 Hz |
| 2005-2006 | AUX_D.xpt | 3 034 | 500-8 000 Hz |
| 2007-2008 | AUX_E.xpt | 1 210 | 500-8 000 Hz |
| 2009-2010 | AUX_F.xpt | 2 368 | 500-8 000 Hz |
| 2011-2012 | AUX_G.xpt | 4 500 | 500-8 000 Hz |
| 2015-2016 | AUX_I.xpt | 4 582 | 500-8 000 Hz |
| 2017-mars 2020 | P_AUX.xpt | 5 147 | 500-8 000 Hz |
| **Total** | | **26 583** | |

### 2.2 Filtres appliqués

| Filtrer | Justification | Avant | Après |
|--------|----------------|-------|--------|
| 20 à 69 ans | Supprimer les cycles avec éligibilité différente (adolescents, 70+) | 26 583 | 14 824 |
| Complétude ≥10/14 | Garantir suffisamment de données par individu | 14 824 | 13 433 |
| ANY25 (≥1 fréquence >25 dB) | Supprimer le « soleil » sain qui a avalé la densité | 13 433 | 7 695 |

### 2.3 Variables confusionnelles connues

| Variables | Impact | Traitement |
|--------------|---------|------------|
| Âge | Fort (R² ~0,57 dans PTA_high) | Filtre 20–69 + centrage des lignes |
| Cycles | Modéré (V de Cramér ~0,16) | Validation par cycle (ARI) |
| Sexe | Faible (V de Cramér ~0,12) | Incontrôlé (futur) |
| 666 (pas de réponse) | 511 lignes (1,9%) | Politique principale : NaN + indicateur |

---

## 3. Prétraitement

### 3.1 Gestion des seuils

| Codes | Signification | Traitement |
|--------|-------------|------------|
| -10 à 120 dB | Valide | Conservé |
| 666 | Pas de réponse (censure sévère) | → NaN + drapeau |
| 888 | Impossible d'obtenir | → NaN |
| Autres | Manquant | → NaN |

### 3.2 Centrage des lignes

Pour chaque individu *i* :

$$\mu_i = \frac{1}{14} \sum_{f \in F} T_{i,f}$$

$$T^{forme}_{i,f} = T_{i,f} - \mu_i$$

Il supprime le « niveau » moyen de perte (combien la personne perd en moyenne) et préserve la « forme » de la courbe (là où la perte est la plus grande/la plus petite).

### 3.3 Mise à l'échelle

RobustScaler (basé sur IQR, quantile_range=(25, 75)). Cela ne suppose pas la normalité. Résistant aux valeurs aberrantes.

### 3.4 Réduction dimensionnelle

PCA avec une variance de 95 % expliquée. Résultat : 10 composants (sur 14).

---

## 4. Modèle

### 4.1 Algorithme

HDBSCAN (Regroupement spatial hiérarchique d'applications avec bruit basé sur la densité).

### 4.2 Hyperparamètres

| Paramètre | Valeur | Justification |
|---------------|-------|---------------|
| min_cluster_size | 10 | Plus petite valeur qui trouve une structure (grille testée : 5–200) |
| min_samples | 5 | HDBSCAN par défaut, testé avec 3–20 |
| métriques | euclidien | Norme pour les données continues |
| méthode_sélection_cluster | eom | Excès de masse (par défaut) |
| core_dist_n_jobs | -1 | Parallélisme |

### 4.3 Grille testée

| min_cluster_size | min_samples | n_clusters | fraction_bruit |
|-----------------|-------------|------------|----------------|
| 5 | 3 | 12 | 0,048 |
| 5 | 5 | 4 | 0,088 |
| **10** | **5** | **2** | **0,076** |
| 15 | 5 | 2 | 0,083 |
| 20 | 5 | 0 | 1 000 |
| 30+ | n'importe quel | 0 | 1 000 |

---

## 5. Résultats

### 5.1 Clusters trouvés

| Grappe | n | %

| Description géométrique |
|---------|---|---|-----------|
| 0 | 7 098 | 92,2% | Perte de pente légère à modérée, bilatérale relativement symétrique |
| 1 | 12 | 0,2% | Asymétrie unilatérale sévère (oreille droite ~80 dB, oreille gauche ~16 dB) |
| Bruit | 585 | 7,6% | Modèles hétérogènes, pertes modérées à sévères |

### 5.2 Métriques

| Métrique | Valeur | Interprétation |
|---------|-------|---------------|
| Bruit HDBSCAN | 7,6% | Faible (était d'environ 90 % avant les filtres) |
| ARI à cycles croisés | 0,27 | Cohérence entre les cycles NHANES avec différentes éligibilités (mesure la robustesse aux variations du protocole/cohorte) |
| Bootstrap ARI (moyen) | 0,68 | Reproductibilité au sein de sous-échantillons de la même population (mesure la stabilité interne) |
| Bootstrap ARI (conditionnel) | 0,60 | Quand les clusters apparaissent (85 % des sous-échantillons) |

> **Remarque :** L'ARI cross-cycle et l'ARI Bootstrap sont des métriques différentes. La première mesure la cohérence entre les différentes populations (cycles NHANES) ; la seconde mesure la reproductibilité au sein d’une même population. Les deux sont signalés par souci de transparence.
| AUC RF (groupe 0 vs 1) | 1.0 | Séparation parfaite (classes déséquilibrées) |

### 5.3 Boîte noire (substitut RF)

Top 7 des caractéristiques discriminantes (toutes de l’oreille droite) :

| Fonctionnalité | Importance de Gini |
|---------|----------------|
| thr_R_1000 | 0,2248 |
| thr_R_500 | 0,2203 |
| thr_R_2000 | 0,1453 |
| thr_R_4000 | 0,1175 |
| thr_R_3000 | 0,1174 |
| thr_R_6000 | 0,0711 |
| thr_R_8000 | 0,0427 |

### 5.4 Acouphènes

| Groupe | Tarif | n valide |
|-------|------|--------------|
| Grappe 0 | 18,3% | 4 397 |
| Groupe 1 | 50,0% | 8 |

> **Remarque :** Le taux d'acouphènes du groupe 1 est basé sur N=8 individus pour lesquels des données sont disponibles. Interpréter comme directionnellement suggestive et non statistiquement définitive.
| Valeurs aberrantes | 38,0% | 308 |

Chi² p<0,001, V de Cramér=0,126.

---

## 6. Analyse de sensibilité (H11)

| Politique | Traitement 666 | ARI contre nan | Impact |
|--------------|----------------|---------------|---------|
| nan (primaire) | 666 → NaN + drapeau | — | Référence |
| cap125 (alternative) | 666 → 125 dB + drapeau | 0,9914 | Minimum |

511 lignes concernées (1,9%). ARI 0,99 dans toutes les politiques → résultat insensible au traitement 666.

### 6.2 Sensibilité du filtre ANY25

| Configuration | N | Grappes | Bruit | ARI vs primaire |
|--------------|---|---------|-------|----------------|
| Avec ANY25 (primaire) | 7 695 | 2 | 7,6% | — |
| Sans ANY25 | 13 433 | 2 | 4,4% | 0,85 |

Le filtre ANY25 supprime le « noyau sain » mais ne déforme pas la structure découverte. 98,9 % des membres du cluster 0 et 75 % des membres du cluster 1 sont conservés dans les paramètres de filtre.

### 6.3 Cohérence du pipeline du OHHR

| OHHRConfiguration | N | Bruit | PTA × SRT r |
|----------|---|-------|---------------|
| Sans ANY25 | 581 | 53,0% | 0,015 |
| Avec ANY25 | 537 | 54,0% | 0,018 |

Le filtre ANY25 appliqué à l'OHHR produit des résultats pratiquement identiques, confirmant que le pipeline est robuste à ce choix.

### 6.4 Stabilité du bootstrap dans l'espace 4D

| Espace | N diminue | ARI médian | Fonctionne avec des clusters | SD |
|-------|--------|------------|---------|-----|
| 14D (seuils complets) | 10 APC | 0,68 | 85% | ~0,40 |
| 4D (moyenne binaurale 500/1k/2k/4k) | 4 APC | **0,74** | **100 %** | **0,016** |

L'espace binaural moyen à 4 fréquences est *plus stable* que l'espace complet à 14 fréquences. Il s’agit de l’espace utilisé pour la validation externe du OHHR, renforçant ainsi la comparaison entre populations.



---

## 7. Validation

### 7.1 Validation par cycle (approximate_predict)

| Cycles | n test | ARI |
|-------|------------|-----|
| 1999-2000 | 949 | 0,17 |
| 2001-2002 | 1 031 | 0,21 |
| 2003-2004 | 1 000 | 0,18 |
| 2011-2012 | 2 238 | 0,37 |
| 2015-2016 | 2 477 | 0,41 |

ARI moyen : 0,27. Les cycles plus récents (N plus élevé) ont un ARI plus élevé.

### 7.2 Bootstrap (100 exécutions × 80 %)

- 85 % des sous-échantillons ont trouvé 2 clusters
- ARI médian : 0,68
- ARI conditionnel (quand 2 clusters) : 0,60
- 15 % d'échec : le cluster 1 (12 personnes) ne se forme pas lors du sous-échantillonnage

### 7,3 Va

relations extérieures — OHHR

**Exécuté.** OHHR (Oldenburg Hearing Health Record ; Jafri et al., 2025) : 581 adultes (âge médian 71 ans, PTA médian 45 dB), CC BY 4.0.

** Pipeline appliqué :**
1. Extraction des 4 fréquences communes avec NHANES (500, 1000, 2000, 4000 Hz)
2. Centrage des lignes (même opération que NHANES)
3. RobustScaler avec paramètres NHANES (non réajusté)
4. Projection sur PCA formé par NHANES (10 composants)
5. `approximate_predict` avec le clusterer NHANES HDBSCAN

**Résultats :**
- 53 % des OHHR sont tombés sous forme de bruit (contre 37,6 % dans la NHANES) – ce qui est attendu car l'OHHR est plus ancien et clinique
- Corrélation PTA × SRT : Pearson r=0,015, Spearman r=−0,007 (N=581)
- Interprétation : l'audiogramme ne prédit pas la parole dans le bruit ("Facteur D")

**Limitation :** L'OHHR ne sépare pas R/L, ce qui rend impossible la comparaison d'asymétrie. Fréquences limitées à 500-4 000 Hz – mais le bootstrap en 4D a montré une stabilité *plus grande* qu'en 14D (ARI 0,74 contre 0,68).

**Cohérence du pipeline :** OHHR avec filtre ANY25 (N=537) : 54,0 % de bruit, PTA×SRT r=0,018 — pratiquement identique à sans filtre (53,0 %, r=0,015).

---

## 8. Limites

### 8.1 Limites des données

1. NHANES est transversal — il n’y a pas de progression temporelle individuelle.
2. NHANES n'a pas d'antécédents de cisplatine pédiatrique - « semblable au platine » est une substitution, pas une confirmation.
3. Fréquences limitées à 500–8 000 Hz — l'ototoxicité peut commencer >8 kHz.
4. Les acouphènes sont autodéclarés (AUQ191) — disponible uniquement dans les cycles de 2005+.
5. Pas de parole dans le bruit — NHANES ne mesure pas la perception fonctionnelle.

### 8.2 Limites du modèle

1. HDBSCAN est sensible à min_cluster_size — les petits clusters peuvent ne pas se former.
2. Le centrage des lignes supprime le niveau – ne capture pas « l'ampleur » de la perte, mais simplement la forme.
3. 14 dimensions sont peu nombreuses, mais elles capturent 95 % de la variance.
4. Le cluster 1 (12 personnes) est trop petit pour une généralisation de la population.
5. L'échec du bootstrap de 15 % montre la sensibilité de l'échantillonnage.

### 8.3 Limites éthiques

1. Aucun cluster n'a reçu une étiquette clinique : il s'agit de géométrie, pas de diagnostic.
2. Le cas personnel du fondateur n'a pas été utilisé dans la formation.
3. Les prévalences ne doivent pas être déduites sans poids d’enquête.
4. Le modèle ne doit pas être utilisé pour des décisions cliniques individuelles.

---

## 9. Utilisation recommandée

| ✅Peut | ❌ Ne doit pas |
|--------------|------------|
| Recherche sur les normes audiométriques | Diagnostic clinique individuel |
| Simulation d'empathie auditive | Inférence de prévalence sans poids |
| Génération d'hypothèses cliniques | Remplacer un audiologiste |
| Validation de cas personnel comme point externe | Utiliser les données personnelles comme base statistique |

---

## 10. Reproductibilité

### 10.1 Environnement

```
Python3.13+
numpy, pandas, scipy, scikit-learn, hdbscan, joblib, plotly
```

### 10.2 Scripts

20 scripts Python, numérotés séquentiellement. Chaque script a des points de contrôle (ne se réexécute pas si la sortie existe).

### 10.3 Données

NHANES XPT public via CDC. URL documentées dans `scripts/00_download_nhanes.py`.

### 10.4 Sorties

Plus de 15 fichiers JSON avec des résultats complets. Le tout reproductible à partir des scripts.

---

## 11. Références

- NHANES : https://wwwn.cdc.gov/nchs/nhanes/
- HDBSCAN : McInnes, L., Healy, J. (2017). Clustering accéléré basé sur la densité hiérarchique. ICDM 2017.
- ARI : Hubert, L., Arabie, P. (1985). Comparaison des partitions. Journal de classification, 2(1), 193-218.

---

## 12. Contacter

La fréquence — gabrielviniciusnascimento345@gmail.com

---

*Carte modèle générée le 2026-05-26. Aucune étiquette clinique n’a été utilisée lors de la formation.*