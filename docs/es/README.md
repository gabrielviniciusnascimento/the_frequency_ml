# The Frequency ML: estado completo del proyecto

**Última actualización:** 2026-05-26  
**Autor:** Gabriel Vinicius Nascimento  
**Contacto:** gabrielviniciusnascimento345@gmail.com  
**Repositorio:** https://github.com/gabrielviniciusnascimento/the_frequency_ml  
**Licencia:** MIT  

---

## ¿Qué es esto?

The Frequency es una herramienta de empatía auditiva basada en la web que permite a las personas con audición normal experimentar cómo les suena el mundo a las personas con pérdida auditiva.

Esta es la capa de ciencia de datos detrás de esto: un proceso de aprendizaje automático que descubre patrones reales de pérdida auditiva en 26,583 personas de NHANES (Encuesta de Salud de EE. UU.), sin imponer etiquetas clínicas como entrada.

El proyecto nació de una condición personal: el autor es un sobreviviente de un hepatoblastoma infantil tratado con cisplatino, con ototoxicidad permanente, ruido, distorsiones y progresión atípica. La experiencia vivida se utiliza como un caso de validación externa, no como una base estadística.

---

## Números que importan

| Métrica | Valor | ¿Qué significa?
|---------|-------|-----------------|
| Audiogramas procesados ​​| **26.583** | Personas reales de NHANES |
| Personas con pérdida auditiva (ANY25) | **7.695** | Subconjunto después de los filtros |
| Clústeres encontrados | **2** | Patrones reales descubiertos por HDBSCAN |
| Ruido | **7,6%** | Antes de los filtros era el 90% |
| Arranque de IRA (mediana) | **0,68** | Reproducible en 85/100 submuestras |
| Interciclos de IRA | **0,27** | Estabilidad moderada |
| Asimetría unilateral (Grupo 1) | **30 personas** | Pérdida severa en 1 oído, el resto normal |
| Tinnitus en los valores atípicos | **38%** | 2 veces más que en el grupo principal |
| Correlación PTA × SRT (OHHR) | **r=0,015** | El audiograma no predice el habla en ambientes ruidosos |
| Secuencias de comandos de Python | **20** | Tubería reproducible |
| Salidas JSON | **15+** | Resultados auditables |
| Panel interactivo | **9 secciones** | Vista completa |

---

## Lo que mostraron los datos (en lenguaje sencillo)

### 1. La mayoría de las pérdidas auditivas son un gradiente, no categorías

No existen "casillas" separadas para el "tipo 1, tipo 2, tipo 3" de pérdida auditiva. Existe una transición suave desde “casi normal” hasta “moderado”. Eso es como decir: no hay 5 tallas de zapato, hay un pie que crece continuamente.

### 2. Hay un grupo real de 30 personas con pérdida severa en un solo oído.

La computadora encontró, sin que nadie se lo dijera, a 30 personas en NHANES que tenían una pérdida severa en el oído derecho y una audición casi normal en el izquierdo. No es un error en los datos: aparece en 4 ciclos diferentes (2001-2016).

### 3. Las personas con pérdida atípica tienen 2 veces más tinnitus

Los 585 casos que no encajan en ningún patrón claro tienen una tasa de tinnitus del 38%, frente al 18% en el grupo principal. Escuchar "extrañeza" se asocia con más síntomas.

### 4. El audiograma no cuenta toda la historia

En el conjunto de datos del OHHR (581 alemanes), la correlación entre el audiograma (PTA) y la capacidad de comprender el habla en ruido (SRT) es prácticamente nula (r=0,015). Las personas con audiogramas similares pueden desempeñarse de manera muy diferente en situaciones reales.

### 5. El sistema de proyección funciona

Cuando colocamos un audiograma hipotético de ototoxicidad por platino en el espacio entrenado, cae en la periferia (percentil 94,9). Cuando ponemos uno normal, cae en el centro (46,8º). El sistema distingue correctamente los patrones.

---

## Lo que se construyó

### Canalización (20 scripts de Python)

| # | Guión | ¿Qué hace?
|---|-----------|-----------|
| 00 | `00_download_nhanes.py` | Descargar datos CDC NHANES |
| 01 | `01_ingest_aux.py` | Armoniza audiogramas (ancho/largo) |
| 02 | `02_merge_context.py` | Combina audiogramas + datos demográficos + cuestionarios |
| 03 | `03_features_v1.py` | Crea 150 funciones derivadas |
| 04 | `04_qa_report.py` | Informe de calidad de datos |
| 05 | `05_h11_sensibilidad_666.py` | Prueba la sensibilidad al código 666 (sin respuesta) |
| 06 | `06_model_ready.py` | Limpia y prepara para modelar |
| 07 | `07_pca_umap.py` | Reducción dimensional + visualización |
| 08 | `08_hdbscan_grid.py` | Búsqueda de cuadrícula HDBSCAN |
| 09 | `09_cluster_profiles.py` | Perfiles geométricos de los clusters |
| 10 |

`10_rf_surrogate.py` | Random Forest para explicar los clusters |
| 11 | `11_generate_results_md.py` | Genera informe de resultados V1 |
| 12 | `12_hdbscan_pca_grid.py` | HDBSCAN en el espacio PCA |
| 13 | `13_kmeans_baseline.py` | KMeans como línea base |
| 14 | `14_artifact_test.py` | Artefactos de las pruebas (edad/ciclo/sexo) |
| 14b | `14b_artifact_per_cluster.py` | Pruebas por grupo individual |
| 15 | `15_residualize_cluster.py` | Elimina el efecto edad/sexo |
| 16 | `16_tinnitus_audit.py` | Auditar tinnitus por ciclo |
| 17 | `17_generate_results_v2_md.py` | Genera informe de resultados V2 |
| 18 | `18_session4_shape_unblock.py` | Sesión 4: ANY25 + centrado de filas |
| 19 | `19_session5_subdivide_cluster0.py` | Subdivisión del grupo principal |
| 20 | `20_session5_outlier_analysis.py` | Análisis de los 585 valores atípicos |
| 21 | `21_session5_rf_surrogate_v2.py` | Sustituto de RF (caja negra) |
| 22 | `22_session5_cluster1_profile.py` | Perfil del 12 con asimetría |
| 23 | `23_session5_tinnitus_clusters.py` | Tinnitus × grupos |
| 24 | `24_session5_personal_projection.py` | Proyección de casos personales |

### Panel interactivo

Archivo HTML autónomo con 9 secciones animadas:
1. El embudo de filtración (26,583 → 7,695)
2. El Espacio Auditivo (PCA coloreado por edad)
3. Los Clústeres (HDBSCAN: 2 + 585 valores atípicos)
4. Audiogramas (mediana por grupo)
5. El 12 (asimetría unilateral individual)
6. Los valores atípicos (distribución de distancias)
7. Lo que separa (importancia de la característica de RF)
8. Tinnitus (por grupo, chi² p<0,001)
9. Bootstrap (100 ejecuciones, mediana de ARI 0,68)

### Documentación

| Archivo | Contenido |
|---------|----------|
| `MODEL_CARD.md` | Tarjeta de plantilla formal (12 secciones) |
| `LITERATURA_REVIEW.md` | 18 referencias, 5 ejes, análisis de brechas |
| `RELATORIO_PROCESSO_COMPLETO.md` | 10 errores documentados, 5 sesiones |
| `RESULTADOS_SESSAO4.md` | Resultados de la sesión 4 |
| `RESULTADOS_SESSAO5.md` | Resultados de la sesión 5 |
| `MAPA_CARREIRA.md` | Oportunidades de financiación y empleo |
| `ANALISE_FINAL_CLAUDE_SESSAO4.md` | Análisis dialéctico entre IA |

### Validación externa

- **OHHR** (Registro de salud auditiva de Oldenburg): 581 personas, CC BY 4.0
  - Habla en ruido (SRT): correlación con PTA ≈ 0
  - Escala de sonoridad disponible
  - Diseñado en el espacio NHANES

---

## Metodología (resumen)

1. **Datos:** NHANES AUX 1999–Mar2020 (9 ciclos, 26,583 personas)
2. **Filtros:** Edad 20–69, integridad ≥10/14, CUALQUIER25 (≥1 frecuencia >25 dB)
3. **Características:** 14 umbrales sin procesar (500–8000 Hz, bilateral)
4. **Preprocesamiento:** Centrado de filas (elimina el nivel, conserva la forma)
5. **Escalado:** RobustScaler (basado en IQR)
6. **Reducción dimensional:** PCA 95 % de variación → 10 componentes
7. **Agrupación:** HDBSCAN (min_cluster_size=10, min_samples=5)
8. **Validación:** Bootstrap 100× (80 % de submuestreo) + ciclos intermedios de ARI
9. **Interpretación:** RF sustituto (500 árboles, class_weight=equilibrado)
10. **Validación externa:** Proyección en la Oficina del Alto Representante (581 personas, Oldenburg)

---

## Limitaciones

1. NHANES es transversal: no existe una progresión temporal individual
2. NHANES no tiene antecedentes de cisplatino pediátrico; "similar al platino" es un sustituto
3. Frecuencias limitadas a 500-8000 Hz: la ototoxicidad puede comenzar >8 kHz
4. El tinnitus lo informan los propios pacientes; solo está disponible en ciclos de más de 2005
5. No se permite hablar en ruido en NHANES: el ACNUDH proporciona parcialmente
6. El grupo 1 (12 personas) es demasiado pequeño para la generalización de la población.
7. 15% de falla de arranque: sensibilidad de muestreo
8. El proyecto no sustituye a audiólogo, otorrinolaringólogo ni oncólogo.

---

## ¿Qué falta?

### Para papel
- [ ] Validación externa con HCHS/SOL o datos clínicos
- [] Revisión completa de la literatura (comenzar hecho)
- [ ] Resumen para conferencia
- [ ] Figuras publicables (alta resolución)

### Para producto
- [ ] Audiogramas personales reales para proyección
- [] Traducción de centroides para filtros DSP
- [] API de proyección audiométrica
- [ ] Traducción a 5 idiomas (EN, ES, PT, DE, FR)

### Para código abierto
- [] README.md profesional con instrucciones
- [] requisitos.txt con dependencia

está arreglado
- [] Pruebas de cordura (3 a 5 pruebas)
- [] Contribuyendo.md

---

## Financiación y oportunidades profesionales

### Inmediato (semanas)
- **Freelance** (Upwork/Fiverr): $50–150/hora en audiología computacional
- **Consultoría** para investigadores: $500–2000/proyecto

### Corto plazo (meses)
- **Microsoft AI para accesibilidad**: entre 5000 y 25 000 dólares + créditos de Azure (continuos, en todo el mundo, su IP)
- **Constructores de Mozilla**: entre 10 000 y 50 000 dólares

### Medio plazo (3–12 meses)
- **NIH R21**: $275,000/2 años (se requiere socio académico)
- **NSF SBIR**: $275 000–1 000 000 (necesita empresa)
- **Empleo en tecnología sanitaria**: entre 80 000 y 150 000 dólares al año

### Largo plazo (más de 12 meses)
- **Frequency freemium**: entre 1000 y 10 000 dólares al mes
- **B2B para clínicas**: $200–1000/mes por clínica
- **Licencia de fabricante**: entre 10 000 y 100 000 dólares al año

---

## Cómo citar este trabajo

```
[Tu nombre]. (2026). The Frequency ML: fenotipado audiométrico basado en datos 
utilizando HDBSCAN en datos NHANES. GitHub. https://github.com/gabrielviniciusnascimento/the_frequency_ml
```

---

## Gracias

- NHANES/CDC para datos públicos
- OHHR/Hearing4all para datos de validación (CC BY 4.0)
- Parthasarathy et al. (2020) para trabajos previos sobre agrupación audiométrica en NHANES
- La comunidad de código abierto para las herramientas (scikit-learn, hdbscan, plotly)

---

## Nota sobre accesibilidad

Este documento está en portugués. Planeamos que esté disponible en 5 idiomas:
- 🇬🇧 Inglés
- 🇬🇧 English
- 🇧🇷 portugués
- 🇩🇪 Alemán
- 🇫🇷 Francés

El panel interactivo es autónomo y funciona en cualquier navegador moderno.

El código es reproducible y documentado. Cada script tiene registros y puntos de control.

---

## Nota sobre el viaje

Este proyecto fue construido por una persona sin una formación completa, en una situación financiera precaria, que aprendió por sí mismo ML, audiología y ciencia de datos, porque necesitaba hacer algo con su propia experiencia como sobreviviente de cáncer infantil con ototoxicidad.

El pipeline que los profesionales de doctorado tardan meses en armar se construyó en 5 sesiones de trabajo. Los resultados son reales, reproducibles y auditables.

La barrera nunca fue técnica. Fue exposición.

Si estás leyendo esto y tienes datos de audiometría, o eres un investigador de la audición, trabajas en tecnología de la salud o eres un sobreviviente como yo, ponte en contacto. El código está abierto. La ciencia está abierta. La puerta está abierta.

---

*Documento generado el 26-05-2026. Todos los datos y hallazgos están disponibles bajo la licencia del MIT.*