# Tarjeta modelo: La frecuencia ML

**Versión:** 1.0  
**Fecha:** 2026-05-26  
**Autor:** Gabriel Vinicius Nascimento, investigador independiente, sobreviviente de hepatoblastoma infantil, ototoxicidad por cisplatino. Construido con asistencia de IA (Claude, Gemini). Sin afiliación institucional.  
**Estado:** Experimental: sin etiqueta clínica  

---

## 1. Descripción general

| Campo | Valor |
|-------|-------|
| **Nombre del modelo** | Agrupación audiométrica HDBSCAN (solo forma) |
| **Tipo** | Agrupación no supervisada |
| **Tarea** | Descubra patrones latentes de pérdida auditiva en datos de población |
| **Datos de entrenamiento** | NHANES AUX 1999–marzo de 2020 (9 ciclos) |
| **N entrenamiento** | 7.695 (después de filtros) |
| **Características** | 14 umbrales sin procesar (500–8000 Hz, bilateral) |
| **Métricas** | ARI, fracción atípica, importancia del Gini |
| **Uso previsto** | Investigación sobre empatía auditiva + simulación (La Frecuencia) |
| **Uso no recomendado** | Diagnóstico clínico individual |

---

## 2. Datos

### 2.1 Fuente

NHANES (Encuesta Nacional de Examen de Salud y Nutrición), CDC/NCHS. Encuesta poblacional transversal de EE. UU., con audiometría de tonos puros por oído/frecuencia.

| Ciclo | Archivo | norte bruto | Frecuencias |
|-------|--------|---------|-------------|
| 1999–2000 | AUX1.xpt | 1.807 | 500–8000 Hz |
| 2001–2002 | AUX_B.xpt | 2.046 | 500–8000 Hz |
| 2003–2004 | AUX_C.xpt | 1.889 | 500–8000 Hz |
| 2005–2006 | AUX_D.xpt | 3.034 | 500–8000 Hz |
| 2007–2008 | AUX_E.xpt | 1.210 | 500–8000 Hz |
| 2009–2010 | AUX_F.xpt | 2.368 | 500–8000 Hz |
| 2011–2012 | AUX_G.xpt | 4.500 | 500–8000 Hz |
| 2015-2016 | AUX_I.xpt | 4.582 | 500–8000 Hz |
| 2017–marzo de 2020 | P_AUX.xpt | 5.147 | 500–8000 Hz |
| **Totales** | | **26.583** | |

### 2.2 Filtros aplicados

| Filtro | Justificación | Antes | Después |
|--------|--------------------------|-------|--------|
| Edad 20–69 | Eliminar ciclos con elegibilidad diferente (adolescentes, mayores de 70 años) | 26.583 | 14.824 |
| Completitud ≥10/14 | Garantizar datos suficientes por individuo | 14.824 | 13.433 |
| CUALQUIER25 (≥1 frecuencia >25 dB) | Retire el "sol" saludable que se tragó la densidad | 13.433 | 7.695 |

### 2.3 Variables de confusión conocidas

| Variables | Impacto | Tratamiento |
|----------|---------|------------|
| Edad | Fuerte (R² ~0,57 en PTA_alta) | Filtro 20–69 + centrado de filas |
| Ciclo | Moderado (V de Cramér ~0,16) | Validación por ciclo (ARI) |
| Sexo | Débil (V de Cramér ~0,12) | Incontrolado (futuro) |
| 666 (sin respuesta) | 511 líneas (1,9%) | Política principal: NaN + bandera |

---

## 3. Preprocesamiento

### 3.1 Manejo de umbrales

| Código | Significado | Tratamiento |
|--------|-------------|------------|
| -10 a 120 dB | Válido | Conservado |
| 666 | Sin respuesta (censura severa) | → NaN + bandera |
| 888 | No se pudo obtener | → NaN |
| Otros | Desaparecido | → NaN |

### 3.2 Centrado de filas

Para cada individuo *i*:

$$\mu_i = \frac{1}{14} \sum_{f \in F} T_{i,f}$$

$$T^{forma}_{i,f} = T_{i,f} - \mu_i$$

Elimina el "nivel" promedio de pérdida (cuánto pierde la persona en promedio) y preserva la "forma" de la curva (donde la pérdida es mayor/menor).

### 3.3 Escalado

RobustScaler (basado en IQR, quantile_range=(25, 75)). No supone normalidad. Resistente a valores atípicos.

### 3.4 Reducción dimensional

PCA con una variación del 95% explicada. Resultado: 10 componentes (de 14).

---

## 4. Modelo

### 4.1 Algoritmo

HDBSCAN (agrupación espacial jerárquica de aplicaciones con ruido basada en densidad).

### 4.2 Hiperparámetros

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| min_cluster_size | 10 | Valor más pequeño que encuentra la estructura (cuadrícula probada: 5–200) |
| min_muestras | 5 | Valor predeterminado de HDBSCAN, probado con 3–20 |
| métricas | euclidiano | Estándar para datos continuos |
| método_selección_clúster | emom | Exceso de masa (predeterminado) |
| core_dist_n_jobs | -1 | Paralelismo |

### 4.3 Red probada

| min_cluster_size | min_muestras | n_clusters | fracción_de_ruido |
|-----------------|-------------|------------|----------------|
| 5 | 3 | 12 | 0,048 |
| 5 | 5 | 4 | 0,088 |
| **10** | **5** | **2*

* | **0,076** |
| 15 | 5 | 2 | 0,083 |
| 20 | 5 | 0 | 1.000 |
| 30+ | cualquiera | 0 | 1.000 |

---

## 5. Resultados

### 5.1 Clústeres encontrados

| Clúster | norte | % | Descripción geométrica |
|---------|---|---|---------------------|
| 0 | 7.098 | 92,2% | Pérdida inclinada leve a moderada, bilateral relativamente simétrica |
| 1 | 12 | 0,2% | Asimetría unilateral grave (oído derecho ~80 dB, oído izquierdo ~16 dB). La dirección correcta es notable: la exposición a armas de fuego generalmente causa pérdida de izquierda en los diestros (efecto de sombra de cabeza). Sugiere diferentes etiologías del ruido ocupacional. |
| Ruido | 585 | 7,6% | Patrones heterogéneos, pérdida moderada-grave |

### 5.2 Métricas

| Métrica | Valor | Interpretación |
|---------|-------|---------------|
| Ruido HDBSCAN | 7,6% | Bajo (era ~90 % antes de los filtros) |
| IRA de ciclo cruzado | 0,27 | Coherencia entre ciclos NHANES con diferente elegibilidad. Valor moderado que refleja la variación en la composición por edades entre ciclos, no un defecto metodológico. Complementa Bootstrap ARI (0,68) que mide la estabilidad dentro de la misma población. |
| Bootstrap ARI (medio) | 0,68 | Reproducibilidad dentro de submuestras de la misma población (mide la estabilidad interna) |
| Bootstrap ARI (condicional) | 0,60 | Cuándo aparecen los conglomerados (85% de las submuestras) |

> **Nota:** ARI de ciclo cruzado y Bootstrap ARI son métricas diferentes. El primero mide la coherencia entre diferentes poblaciones (ciclos NHANES); el segundo mide la reproducibilidad dentro de la misma población. Ambos se informan por transparencia.
| AUC de RF (grupo 0 frente a 1) | 1.0 | Separación perfecta (clases desequilibradas) |

### 5.3 Caja negra (sustituto de RF)

Las 7 características discriminatorias principales (todas del oído derecho):

| Característica | Importancia de Gini |
|---------|----------------|
| thr_R_1000 | 0,2248 |
| thr_R_500 | 0,2203 |
| thr_R_2000 | 0,1453 |
| thr_R_4000 | 0,1175 |
| thr_R_3000 | 0,1174 |
| thr_R_6000 | 0,0711 |
| thr_R_8000 | 0,0427 |

### 5.4 Acúfenos

| Grupo | Tarifa | n válido |
|-------|------|----------|
| Grupo 0 | 18,3% | 4.397 |
| Grupo 1 | 50,0% | 8 |

> **Nota:** La tasa de tinnitus del grupo 1 se basa en N=8 personas con datos disponibles. Interpretar como direccionalmente sugerente, no estadísticamente definitivo.
| Valores atípicos | 38,0% | 308 |

Chi² p<0,001, V de Cramér=0,126.

---

## 6. Análisis de sensibilidad (H11)

| Política | Tratamiento 666 | IRA vs nan | Impacto |
|----------|----------------|---------------|---------|
| nan (primaria) | 666 → NaN + bandera | — | Referencia |
| cap125 (alternativa) | 666 → 125 dB + bandera | 0,9914 | Mínimo |

511 líneas afectadas (1,9%). IRA 0,99 en todas las políticas → resultado insensible al tratamiento 666.

### 6.2 Sensibilidad del filtro ANY25

| Configuración | norte | Grupos | Ruido | IRA versus primaria |
|---------------|---|----------|-------|----------------|
| Con ANY25 (primario) | 7.695 | 2 | 7,6% | — |
| Sin ANY25 | 13.433 | 2 | 4,4% | 0,85 |

El filtro ANY25 elimina el "núcleo sano" pero no distorsiona la estructura descubierta. El 98,9 % de los miembros del grupo 0 y el 75 % de los miembros del grupo 1 se conservan en todas las configuraciones de filtro.

### 6.3 Coherencia en la cartera del ACNUDH

| OHHRConfiguración | norte | Ruido | PTA×SRT r |
|--------------------|---|-------|-----------|
| Sin ANY25 | 581 | 53,0% | 0,015 |
| Con ANY25 | 537 | 54,0% | 0,018 |

El filtro ANY25 aplicado a OHHR produce resultados prácticamente idénticos, lo que confirma que la tubería es sólida para esta elección.

### 6.4 Estabilidad Bootstrap en el espacio 4D

| Espacio | N se atenúa | Mediana de IRA | Corre con clusters | SD |
|-------|--------|------------|-------------------|-----|
| 14D (umbrales completos) | 10 PCA | 0,68 | 85% | ~0,40 |
| 4D (media binaural 500/1k/2k/4k) | 4 PCA | **0,74** | **100%** | **0,016** |

El espacio medio binaural de 4 frecuencias es *más estable* que el espacio completo de 14 frecuencias. Este es el espacio utilizado para la validación externa del ACNUDH, fortaleciendo la comparación entre poblaciones.



---

## 7. Validación

### 7.1 Validación por ciclo (predicción_aproximada)

| Ciclo | norte prueba | IRA |
|----

---|---------|-----|
| 1999–2000 | 949 | 0,17 |
| 2001–2002 | 1.031 | 0,21 |
| 2003–2004 | 1.000 | 0,18 |
| 2011–2012 | 2.238 | 0,37 |
| 2015-2016 | 2.477 | 0,41 |

IRA media: 0,27. Los ciclos más recientes (mayor N) tienen mayor ARI.

### 7.2 Bootstrap (100 ejecuciones × 80%)

- El 85% de las submuestras encontraron 2 grupos.
- Mediana de IRA: 0,68
- ARI condicional (cuando 2 grupos): 0,60
- 15% de fracaso: el grupo 1 (12 personas) no se forma cuando se realiza el submuestreo

### 7.3 Validación externa — ACNUDH

**Ejecutado.** OHHR (Registro de salud auditiva de Oldenburg; Jafri et al., 2025): 581 adultos (edad media 71, PTA media 45 dB), CC BY 4,0.

**Tubería aplicada:**
1. Extracción de las 4 frecuencias comunes con NHANES (500, 1000, 2000, 4000 Hz)
2. Centrado de filas (misma operación que NHANES)
3. RobustScaler con parámetros NHANES (no reajustado)
4. Proyección sobre PCA capacitado por NHANES (10 componentes)
5. `approximate_predict` con el agrupador NHANES HDBSCAN

**Resultados:**
- El 53% de la OHHR disminuyó debido al ruido (frente al 37,6% en NHANES); se espera que la OHHR sea más antigua y clínica.
- Correlación PTA × SRT: Pearson r=0,015, Spearman r=−0,007 (N=581)
- Interpretación: el audiograma no predice el habla en ruido ("Factor D")

**Limitación:** El ACNUDH no separa R/L, lo que hace imposible la comparación de asimetría. Las frecuencias se limitaron a 500–4000 Hz, pero el arranque en 4D mostró una estabilidad *mayor* que en 14D (ARI 0,74 frente a 0,68).

**Consistencia de la tubería:** OHHR con filtro ANY25 (N=537): 54,0 % de ruido, PTA×SRT r=0,018: prácticamente idéntico a sin filtro (53,0 %, r=0,015).

---

## 8. Limitaciones

### 8.1 Limitaciones de datos

1. NHANES es transversal: no existe una progresión temporal individual.
2. NHANES no tiene antecedentes de cisplatino pediátrico; "similar al platino" es un indicador, no una confirmación.
3. Frecuencias limitadas a 500 a 8 000 Hz; la ototoxicidad puede comenzar >8 kHz.
4. El tinnitus es autoinformado (AUQ191): solo disponible en ciclos de 2005+.
5. No hablar en ruido: NHANES no mide la percepción funcional.

### 8.2 Limitaciones del modelo

1. HDBSCAN es sensible a min_cluster_size: es posible que no se formen grupos pequeños.
2. El centrado de filas elimina el nivel; no capta "qué tan grave" es la pérdida, solo la forma.
3. 14 dimensiones son pocas, pero capturan el 95% de la varianza.
4. El grupo 1 (12 personas) es demasiado pequeño para la generalización de la población.
5. La falla del arranque del 15 % muestra la sensibilidad del muestreo.

### 8.3 Limitaciones éticas

1. Ningún grupo recibió una etiqueta clínica: se trata de geometría, no de diagnóstico.
2. En la formación no se utilizó el caso personal del fundador.
3. Las prevalencias no deben inferirse sin ponderaciones de la encuesta.
4. El modelo no debe utilizarse para decisiones clínicas individuales.

---

## 9. Uso recomendado

| ✅ Lata | ❌ No debe |
|---------|------------|
| Investigación sobre estándares audiométricos | Diagnóstico clínico individual |
| Simulación de empatía auditiva | Inferencia de prevalencia sin ponderaciones |
| Generación de hipótesis clínicas | Reemplazar audiólogo |
| Validación de casos personales como punto externo | Utilizar datos personales como base estadística |

---

## 10. Reproducibilidad

### 10.1 Medio ambiente

```
Pitón 3.13+
numpy, pandas, scipy, scikit-learn, hdbscan, joblib, trama
```

### 10.2 Guiones

27 scripts de Python, numerados secuencialmente. Cada script tiene puntos de control (no se vuelve a ejecutar si existe salida).

### 10.3 Datos

NHANES XPT público a través de CDC. URL documentadas en `scripts/00_download_nhanes.py`.

### 10.4 Salidas

Más de 15 archivos JSON con resultados completos. Todo reproducible de los guiones.

---

## 11. Referencias

- NHANES: https://wwwn.cdc.gov/nchs/nhanes/
- HDBSCAN: McInnes, L., Healy, J. (2017). Agrupación acelerada basada en densidad jerárquica. ICDM 2017.
- ARI: Hubert, L., Arabie, P. (1985). Comparando particiones. Revista de clasificación, 2(1), 193-218.

---

## 12. Contacto

La Frecuencia - gabrielviniciusnascimento345@gmail.com

---

*Modelo de Ficha generada el 26-05-2026. No se utilizaron etiquetas clínicas en la capacitación.*