# La fréquence ML - État complet du projet

**Dernière mise à jour :** 2026-05-26  
**Auteur :** Gabriel Vinicius Nascimento  
**Contact :** gabrielviniciusnascimento345@gmail.com  
**Dépôt :** https://github.com/gabrielviniciusnascimento/the_ Frequency_ml  
**Licence :** MIT  

---

## Qu'est-ce que c'est ?

The Frequency est un outil d'empathie auditive basé sur le Web qui permet aux personnes ayant une audition normale de découvrir à quoi ressemble le monde pour les personnes malentendantes.

Il s'agit de la couche de science des données qui se cache derrière : un pipeline d'apprentissage automatique qui découvre des modèles réels de perte auditive chez 26 583 personnes de la NHANES (US Health Survey), sans imposer d'étiquettes cliniques en entrée.

Le projet est né d'une condition personnelle : l'auteur est un survivant d'un hépatoblastome infantile traité au cisplatine, avec une ototoxicité permanente, du bruit, des distorsions et une progression atypique. L’expérience vécue est utilisée comme cas de validation externe et non comme base statistique.

---

## Des chiffres qui comptent

| Métrique | Valeur | Qu'est-ce que cela signifie |
|---------|-------|-----------------|
| Audiogrammes traités | **26 583** | De vraies personnes NHANES |
| Personnes malentendantes (ANY25) | **7 695** | Sous-ensemble après filtres |
| Clusters trouvés | **2** | Modèles réels découverts par HDBSCAN |
| Bruit | **7,6 %** | C'était 90% avant les filtres |
| bootstrap ARI (médiane) | **0,68** | Reproductible en sous-échantillons 85/100 |
| Intercycles ARI | **0,27** | Stabilité modérée |
| Asymétrie unilatérale (Cluster 1) | **30 personnes** | Perte sévère d'une oreille, autre normale |
| Acouphènes dans les valeurs aberrantes | **38%** | 2x plus que dans le groupe principal |
| Corrélation PTA × SRT (OHHR) | **r=0,015** | L'audiogramme ne prédit pas la parole dans le bruit |
| Scripts Python | **20** | Pipeline reproductible |
| Sorties JSON | **15+** | Résultats vérifiables |
| Tableau de bord interactif | **9 rubriques** | Vue complète |

---

## Ce que les données ont montré (en langage clair)

### 1. La plupart des pertes auditives sont un gradient et non des catégories

Il n’existe pas de « cases » distinctes de « type 1, type 2, type 3 » de perte auditive. Il existe un continuum fluide allant de « proche de la normale » à « modéré ». C'est comme dire : il n'y a pas 5 tailles de chaussures, il y a un pied qui grandit continuellement.

### 2. Il existe un groupe réel de 30 personnes avec une perte sévère d'une seule oreille

L'ordinateur a trouvé, sans que personne ne le leur dise, 30 personnes à NHANES qui ont une perte sévère de l'oreille droite et une audition presque normale à la gauche. Il ne s’agit pas d’une erreur dans les données : elle apparaît dans 4 cycles différents (2001-2016).

### 3. Les personnes ayant une perte atypique ont 2 fois plus d'acouphènes

Les 585 cas qui ne correspondent à aucune tendance claire ont un taux d'acouphènes de 38 %, contre 18 % dans le groupe principal. Entendre « l’étrangeté » est associé à davantage de symptômes.

### 4. L'audiogramme ne raconte pas toute l'histoire

Dans l’ensemble de données de l’OHHR (581 Allemands), la corrélation entre l’audiogramme (PTA) et la capacité à comprendre la parole dans le bruit (SRT) est pratiquement nulle (r=0,015). Les personnes ayant des audiogrammes similaires peuvent se comporter de manière très différente dans des situations réelles.

### 5. Le système de projection fonctionne

Lorsque nous plaçons un hypothétique audiogramme d’ototoxicité du platine dans l’espace entraîné, il se situe en périphérie (94,9e percentile). Quand on en met un normal, il tombe au centre (46,8º). Le système distingue correctement les modèles.

---

## Ce qui a été construit

### Pipeline (27 scripts Python)

| # | Scénario | Qu'est-ce que ça fait |
|---|-----------|---------------|
| 00 | `00_download_nhanes.py` | Télécharger les données du CDC NHANES |
| 01 | `01_ingest_aux.py` | Harmonise les audiogrammes (large/long) |
| 02 | `02_merge_context.py` | Combine audiogrammes + données démographiques + questionnaires |
| 03 | `03_features_v1.py` | Crée 150 fonctionnalités dérivées |
| 04 | `04_qa_report.py` | Rapport sur la qualité des données |
| 05 | `05_h11_sensitivity_666.py` | Tests de sensibilité au code 666 (pas de réponse) |
| 06 | `06_model_ready.py` | Nettoie et prépare au modelage |
| 07 | `07_pca_umap.py` | Réduction dimensionnelle + visualisation |
| 08 | `08_hdbscan_grid.py` | Gri

d recherche à partir de HDBSCAN |
| 09 | `09_cluster_profiles.py` | Profils géométriques des clusters |
| 10 | `10_rf_surrogate.py` | Random Forest pour expliquer les clusters |
| 11 | `11_generate_results_md.py` | Génère le rapport de résultats V1 |
| 12 | `12_hdbscan_pca_grid.py` | HDBSCAN dans l'espace PCA |
| 13 | `13_kmeans_baseline.py` | KMeans comme référence |
| 14 | `14_artifact_test.py` | Artefacts de tests (âge/cycle/sexe) |
| 14b | `14b_artifact_per_cluster.py` | Tests par cluster individuel |
| 15 | `15_residualize_cluster.py` | Supprime l'effet âge/sexe |
| 16 | `16_tinnitus_audit.py` | Auditer les acouphènes par cycle |
| 17 | `17_generate_results_v2_md.py` | Génère un rapport de résultats V2 |
| 18 | `18_session4_shape_unblock.py` | Session 4 : ANY25 + centrage des lignes |
| 19 | `19_session5_subdivide_cluster0.py` | Subdivision du cluster principal |
| 20 | `20_session5_outlier_analysis.py` | Analyse des 585 valeurs aberrantes |
| 21 | `21_session5_rf_surrogate_v2.py` | Substitut RF (boîte noire) |
| 22 | `22_session5_cluster1_profile.py` | Profil du 12 avec asymétrie |
| 23 | `23_session5_tinnitus_clusters.py` | Acouphènes × grappes |
| 24 | `24_session5_personal_projection.py` | Projection de cas personnel |

### Tableau de bord interactif

Fichier HTML autonome avec 9 sections animées :
1. L'entonnoir filtrant (26 583 → 7 695)
2. L'Espace Auditif (PCA coloré selon l'âge)
3. Les Clusters (HDBSCAN : 2 + 585 valeurs aberrantes)
4. Audiogrammes (médiane par cluster)
5. Le 12 (asymétrie individuelle unilatérale)
6. Les valeurs aberrantes (distribution des distances)
7. Ce qui sépare (importance des fonctionnalités RF)
8. Acouphènes (par groupe, chi² p<0,001)
9. Bootstrap (100 exécutions, ARI médian 0,68)

###Documentations

| Archives | Contenu |
|---------|----------|
| `MODEL_CARD.md` | Carte modèle formelle (12 sections) |
| `LITERATURA_REVIEW.md` | 18 références, 5 axes, analyse des écarts |
| `RELATORIO_PROCESSO_COMPLETO.md` | 10 erreurs documentées, 5 sessions |
| `RESULTADOS_SESSAO4.md` | Résultats de la séance 4 |
| `RESULTADOS_SESSAO5.md` | Résultats de la séance 5 |
| `MAPA_CARREIRA.md` | Possibilités de financement et d'emploi |
| `ANALISE_FINAL_CLAUDE_SESSAO4.md` | Analyse dialectique entre IA |

### Validation externe

- **OHHR** (Oldenburg Hearing Health Record) : 581 personnes, CC BY 4.0
  - Parole dans le bruit (SRT) : corrélation avec PTA ≈ 0
  - Mise à l'échelle du volume disponible
  - Conçu dans l'espace NHANES

---

## Méthodologie (résumé)

1. **Données :** NHANES AUX 1999–mars 2020 (9 cycles, 26 583 personnes)
2. **Filtres :** Âge 20-69 ans, exhaustivité ≥10/14, ANY25 (≥1 fréquence >25 dB)
3. **Caractéristiques :** 14 seuils bruts (500–8 000 Hz, bilatéraux)
4. **Prétraitement :** Centrage des lignes (supprime le niveau, préserve la forme)
5. **Mise à l'échelle :** RobustScaler (basé sur IQR)
6. **Réduction dimensionnelle :** Variation PCA 95 % → 10 composants
7. **Regroupement :** HDBSCAN (min_cluster_size=10, min_samples=5)
8. **Validation :** Bootstrap 100× (sous-échantillonnage à 80 %) + inter-cycles ARI
9. **Interprétation :** Substitut RF (500 arbres, class_weight=balanced)
10. **Validation externe :** Projection à l'OHHR (581 personnes, Oldenburg)

---

## Limites

1. NHANES est transversal — il n’y a pas de progression temporelle individuelle
2. NHANES n’a pas d’antécédents de cisplatine pédiatrique – « de type platine » est un proxy
3. Fréquences limitées à 500-8 000 Hz — l'ototoxicité peut commencer > 8 kHz
4. Les acouphènes sont autodéclarés – disponibles uniquement dans les cycles de plus de 2005
5. Pas de parole dans le bruit dans la NHANES — l’OHHR fournit en partie
6. Le cluster 1 (12 personnes) est trop petit pour une généralisation de la population
7. Échec du bootstrap de 15 % — sensibilité d'échantillonnage
8. Le projet ne remplace pas l'audiologiste, l'oto-rhino-laryngologie ou l'oncologie

---

## Ce qui manque

### Pour le papier
-[ ] Validation externe avec HCHS/SOL ou données cliniques
- [ ] Revue complète de la littérature (début terminé)
- [ ] Résumé pour la conférence
- [ ] Chiffres publiables (haute résolution)

### Pour le produit
- [ ] De vrais audiogrammes personnels pour la projection
-[ ] Traduction des centroïdes pour les filtres DSP
- [ ] API de projection audiométrique
- [ ] Traduction en 5 langues (EN, ES, PT, DE, FR)

### À

source ouverte
- [ ] README.md professionnel avec instructions
- [ ] Requirements.txt avec dépendances corrigées
- [ ] Tests de santé mentale (3 à 5 tests)
- [ ] Contribuer.md

---

## Financement et opportunités de carrière

### Immédiat (semaines)
- **Freelance** (Upwork/Fiverr) : 50 à 150 $/heure en audiologie computationnelle
- **Conseil** pour les chercheurs : 500 à 2 000 $/projet

### Court terme (mois)
- **Microsoft AI for Accessibility** : 5 000 à 25 000 $ + crédits Azure (en continu, dans le monde entier, votre IP)
- **Mozilla Builders** : 10 000 à 50 000 $

### Moyen terme (3 à 12 mois)
- **NIH R21** : 275 000 $/2 ans (partenaire académique requis)
- **NSF SBIR** : 275 000 à 1 000 000 $ (entreprise nécessaire)
- **Emploi dans les technologies de la santé** : 80 000 à 150 000 $/an

### Longue durée (12+ mois)
- **La fréquence freemium** : 1 000 à 10 000 $/mois
- **B2B pour les cliniques** : 200 à 1 000 $/mois par clinique
- **Licence de fabricant** : 10 000 à 100 000 $/an

---

## Comment citer cet ouvrage

```
Gabriel Vinicius Nascimento. (2026). The Frequency ML : phénotypage audiométrique basé sur les données 
en utilisant HDBSCAN sur les données NHANES. GitHub. https://github.com/gabrielviniciusnascimento/the_ Frequency_ml
```

---

## Merci

- NHANES/CDC pour les données publiques
- OHHR/Hearing4all pour les données de validation (CC BY 4.0)
-Parthasarathy et al. (2020) pour des travaux antérieurs sur le clustering audiométrique dans NHANES
- La communauté open-source pour les outils (scikit-learn, hdbscan, plotly)

---

## Remarque sur l'accessibilité

Ce document est en portugais. Nous prévoyons de le rendre disponible en 5 langues :
- 🇫🇷 Anglais
- 🇪🇸 Español
- 🇧🇷 Portugais
- 🇩🇪 Allemand
- 🇫🇷 English

Le tableau de bord interactif est autonome et fonctionne dans n'importe quel navigateur moderne.

Le code est reproductible et documenté. Chaque script a une journalisation et des points de contrôle.

---

## Remarque sur le voyage

Ce projet a été construit par une personne sans formation complète, dans une situation financière précaire, qui a appris lui-même le ML, l'audiologie et la science des données — parce qu'il avait besoin de faire quelque chose de sa propre expérience de survivant d'un cancer infantile avec ototoxicité.

Le pipeline que les professionnels doctorants mettent des mois à constituer a été construit en 5 séances de travail. Les résultats sont réels, reproductibles et vérifiables.

La barrière n’a jamais été technique. C'était une exposition.

Si vous lisez ceci et disposez de données d'audiométrie, si vous êtes un chercheur en audition, si vous travaillez dans le domaine des technologies de la santé ou si vous êtes un survivant comme moi, contactez-nous. Le code est ouvert. La science est ouverte. La porte est ouverte.

---

*Document généré le 2026-05-26. Toutes les données et résultats sont disponibles sous licence MIT.*