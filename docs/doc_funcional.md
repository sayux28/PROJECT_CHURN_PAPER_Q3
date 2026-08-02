# Documentación técnica del proyecto

## Resumen del pipeline de preprocesamiento (checkpoint antes de modelar)

Todo lo aplicado hasta ahora, en orden (`scripts/eda_churn_v1.py` + `scripts/eda_p1.py`):

1. **Carga:** dataset Maven Analytics vía kagglehub (o CSV local) — 7043 clientes, 38 columnas + población por zip (no se usa como feature).
2. **Diagnósticos sobre el dataset completo:** `Monthly Charge` negativo (120 casos, detectados no corregidos aún) y Tenure por `Customer Status` (boxplot que confirma 503 `Churned` con ≤2 meses de antigüedad).
3. **Target:** `build_target()` excluye a los 454 clientes `Joined` (censura — ver sección más abajo) y crea `Churn_Flag`. Quedan 6589 clientes.
4. **Split:** 80/20 estratificado por `Churn_Flag`, semilla fija (42) — replica el protocolo del paper SHAP-Rule (Sección 4.3). Train: 5271 | Test: 1318.
5. **Imputación de nulos — dos tipos:**
   - *Estructural* (`imputar_estructural()`): nulos que no son datos perdidos sino consecuencia del diccionario de datos (ej. sin internet → columnas de servicios de internet vacías). Se rellenan con reglas fijas (`"No Internet Service"`, `"No Phone Service"`, `0`, `"None"`), no con estadísticos — no hay riesgo de leakage.
   - *Estadística* (`corregir_monthly_charge()`): los 120 valores negativos de `Monthly Charge` se reemplazan por la mediana calculada **solo con train** (73.15), aplicada después a train y test.
   - Resultado: 0 nulos tras estos dos pasos.
6. **Análisis de numéricas (12 variables):** varianza casi nula (`Total Refunds`), sesgo, outliers IQR (diagnóstico, nada se elimina), correlación + VIF → se eliminaron `Total Charges` y `Total Revenue` por multicolinealidad severa (identidad exacta `Total Revenue = Total Charges − Total Refunds + Total Long Distance Charges + Total Extra Data Charges`). Quedan 10 numéricas.
7. **Categóricas (18 variables):** codificación ordinal (un entero por categoría, fit solo en train) — no one-hot, por la estrategia SHAP-Rule.
8. **Matriz final:** 10 numéricas (sin escalar) + 18 categóricas (ordinal) = 28 columnas, guardadas en `data/processed/`.

**Lo que NO se hizo, a propósito:** ni `StandardScaler` ni `MinMaxScaler` (los árboles no lo necesitan), ni PCA (rompe la trazabilidad variable→regla de SHAP-Rule), ni eliminación de filas por outliers, ni balanceo de clases todavía (ver más abajo).

## Balanceo de clases: por qué no se usa SMOTE

El target queda desbalanceado 28.4% churn / 71.6% no-churn (train) — un desbalance **moderado**, no severo (severo sería <10%).

**Por qué no SMOTE:** SMOTE es una técnica de *resampling* — genera filas sintéticas nuevas interpolando entre clientes vecinos, para que ambas clases queden con cantidades parecidas antes de entrenar. Dos problemas concretos para este proyecto:

1. **Choca con la codificación ordinal.** Las categóricas están codificadas como enteros (`Internet Type`: Cable=0, DSL=1, Fiber Optic=2...). SMOTE interpola entre vecinos, así que puede generar un cliente sintético con `Internet Type = 1.4` — un valor que no corresponde a ninguna categoría real. Para SHAP-Rule esto es especialmente delicado: una regla del tipo "SI Internet Type ≈ 1.4..." no se traduce a ningún antecedente lingüístico interpretable.
2. **El desbalance no es lo bastante severo como para justificarlo.** Con 28.4%/71.6%, los ensambles de árboles suelen aprender bien la clase minoritaria sin generar datos artificiales, sobre todo reportando métricas ponderadas (recall/precision/F1), como ya contempla la Fase 4 del plan.

**Qué se usa en su lugar: `class_weight` / `scale_pos_weight` (cost-sensitive learning, no resampling).** La diferencia de fondo:

- **SMOTE** actúa sobre los **datos**: crea filas nuevas antes de entrenar.
- **`class_weight='balanced'`** (Random Forest) o **`scale_pos_weight`** (XGBoost/LightGBM, ≈ negativos/positivos ≈ 2.5) actúan sobre el **algoritmo**: no se toca ni una fila del dataset — el train sigue siendo 28.4%/71.6% tal cual. Lo que cambia es cuánto "cuesta" para el algoritmo equivocarse en cada clase durante el entrenamiento (un error en un cliente que sí hizo churn pesa ~2.5 veces más que uno en un cliente que no), y el modelo ajusta sus splits/pesos internos en consecuencia.

Esta opción evita por completo el problema de interpolación ordinal de SMOTE (no genera ningún cliente nuevo) y se integra directo como un hiperparámetro más dentro del `GridSearchCV` ya planeado en la Fase 4.

### LightGBM: `class_weight` vs. `scale_pos_weight` — se elige `scale_pos_weight`

Para Random Forest no hay nada que decidir: `RandomForestClassifier` solo acepta `class_weight`. Para LightGBM (y XGBoost) sí existen **dos formas distintas de pedir lo mismo**, y no se deben combinar las dos a la vez porque sus efectos se superponen de forma poco predecible (la propia documentación de LightGBM lo advierte):

| Opción | Cómo funciona | Cuándo usarla |
|---|---|---|
| `class_weight='balanced'` (o `is_unbalance=True`) | Calcula el peso de cada clase automáticamente, a partir de las frecuencias observadas en el train | Cuando no se quiere/puede calcular el ratio a mano, o el desbalance puede variar entre corridas |
| `scale_pos_weight=<ratio>` | Un único número fijo: `negativos / positivos` | Cuando el ratio ya se conoce y se quiere control explícito y reproducible sobre cuánto pesa la clase positiva |

**Se eligió `scale_pos_weight`**, con el ratio calculado directamente del train (`(y_train == 0).sum() / (y_train == 1).sum()`, ≈2.5). Razones:

1. **Es más preciso para el caso binario** — `scale_pos_weight` multiplica el gradiente de la clase positiva por ese factor exacto; `class_weight='balanced'` hace un cálculo más genérico pensado también para multiclase, que en el caso binario termina siendo equivalente pero menos directo de auditar.
2. **Es reproducible y queda documentado como un número concreto** (≈2.5) en vez de una fórmula interna que recalcula sola — más fácil de explicar y defender en la exposición.
3. Es la opción que la propia documentación de LightGBM recomienda específicamente para clasificación binaria desbalanceada, dejando `is_unbalance`/`class_weight` como alternativa cuando no se conoce el ratio de antemano.

En `eda_p1.py`, Random Forest usa `class_weight='balanced'` y LightGBM usa `scale_pos_weight` — nunca ambas técnicas juntas en el mismo modelo.

## Preprocesamiento: dos pipelines según el tipo de modelo

A partir del mismo dataset base (`telecom_customer_churn.csv`, Maven Analytics vía kagglehub) y el mismo preprocesamiento inicial (target, split train/test, imputación estructural, corrección de `Monthly Charge` negativo — todo en `scripts/eda_churn_v1.py`), las variables categóricas se convierten a numérico de dos formas distintas, según el tipo de modelo que va a consumir la salida. `scripts/eda_p1.py` (proyecto SHAP-Rule completo) solo cubre la primera fila; `scripts/eda_p1_ml.py` corre el preprocesamiento común una sola vez y luego genera las dos salidas para comparar los 4 modelos (Árbol de Decisión, Random Forest, LightGBM y Regresión Logística):

| Encoding categóricas | Numéricas | Salida | Para qué |
|---|---|---|---|
| Ordinal (un entero por categoría, una sola columna por variable) | Sin escalar | `data/processed/` | Random Forest / XGBoost / LightGBM (proyecto SHAP-Rule) |
| One-Hot (`drop='if_binary'`) | `StandardScaler` | `data/processed_linear/` | Probar un modelo lineal (regresión logística, SVM) para ver si los datos se comportan de forma linealmente separable |

**Por qué la diferencia:** los ensambles de árboles (RF, XGBoost/LightGBM) dividen por umbrales columna por columna y son invariantes a la escala y al valor numérico asignado a cada categoría, así que no necesitan One-Hot ni escalado — y con codificación ordinal cada variable original conserva un único valor SHAP, lo que simplifica traducirlo a reglas lingüísticas (ver `docs/plan_shap_rule_churn_telecom.md`). Un modelo lineal, en cambio, sí interpreta la magnitud de los números: con ordinal asumiría relaciones de orden que no existen (ej. que "Fiber Optic" vale el doble que "DSL"), por eso ahí se usa One-Hot + escalado.

Las dos salidas se guardan en carpetas distintas (`data/processed/` vs `data/processed_linear/`) para que no se sobreescriban entre sí.

## Por qué se excluyen los clientes "Joined" del modelado

`Customer Status` tiene 3 valores, no 2: `Stayed`, `Churned` y `Joined`. `Joined` son clientes dados de alta **este trimestre** — verificado con los datos reales, su `Tenure in Months` va de 1 a 3 meses únicamente (mín=1, mediana=1, máx=3), sin excepción. `build_target()` (en `scripts/eda_churn_v1.py`) los excluye del entrenamiento antes del split; `eda_p1.py` hereda esa exclusión.

### La objeción obvia, y por qué no aplica

Se podría pensar: "si un `Joined` todavía no se fue, es razonable etiquetarlo como no-churn (`Churn_Flag=0`)". El problema es que **el dataset es una foto tomada en un momento fijo**, y para un `Joined` esa foto se tomó demasiado pronto para saber su desenlace real — a diferencia de `Stayed`/`Churned`, que sí tienen su historia cerrada.

### La evidencia que descarta la objeción

Se generó `docs/tenure_por_status.png` (boxplot de `Tenure in Months` por `Customer Status`, función `graficar_tenure_por_status()` en `eda_p1.py`) sobre el dataset completo, antes de excluir nada:

| Customer Status | mín | 25% | mediana | 75% | máx |
|---|---|---|---|---|---|
| `Joined` | 1 | 1 | 1 | 2 | 3 |
| `Churned` | **1** | 2 | 10 | 29 | 72 |
| `Stayed` | **4** | 21 | 42 | 63 | 72 |

El bigote inferior de `Churned` llega hasta 1 mes — se solapa por completo con el rango de `Joined` (1-3 meses). En números: **503 clientes `Churned` tienen `Tenure <= 2 meses`**. Es decir, el churn temprano no es un caso raro — es frecuente.

**Esto invierte el argumento, no lo confirma.** Si el churn temprano fuera raro, forzar `Joined → 0` sería una aproximación de bajo riesgo. Pero como es frecuente, es estadísticamente probable que una fracción de los `Joined` — indistinguibles hoy de los `Churned` tempranos en cuanto a antigüedad — terminen en `Churned` en el próximo corte de datos. Etiquetarlos igual a todos como "0" fabrica una etiqueta incorrecta para un número desconocido de ellos.

### El concepto: censura (censoring)

Es un problema conocido en modelado de supervivencia/churn: un caso está **censurado** cuando su desenlace final todavía no se observó al momento de cortar los datos. La práctica estándar es excluir los casos censurados del entrenamiento supervisado en vez de forzarles una etiqueta asumida — exactamente lo que hace `build_target()`.

### Experimento de comparación: eda_p2.py

`scripts/eda_p2.py` es una copia de `eda_p1.py` que hace **lo contrario a propósito**: no excluye a `Joined`, los incluye con `Churn_Flag=0` (mismo tratamiento que `Stayed`). No es el enfoque recomendado — existe para poder comparar en la práctica, entrenando el mismo tipo de modelo sobre ambas salidas, cuánto cambia la precisión al mezclar casos censurados como si fueran "no-churn" confirmado. Guarda su salida en `data/processed_incluye_joined/` y su log en `docs/hallazgos_numericas_p2.txt`, para no pisar los archivos de `eda_p1.py`.

## Estandarización vs. Normalización (StandardScaler vs. MinMaxScaler)

"Normalizar" se usa mucho como palabra genérica para "llevar los datos a una escala comparable", pero **estandarización** y **normalización** son dos fórmulas distintas, con dos resultados distintos. `StandardScaler` (el que usa la rama de regresión logística de `eda_p1_ml.py`) hace **estandarización**, no normalización.

### Las dos fórmulas

| | Estandarización (`StandardScaler`) | Normalización (`MinMaxScaler`) |
|---|---|---|
| Fórmula | `(x − media) / desviación_estándar` | `(x − mínimo) / (máximo − mínimo)` |
| Usa de los datos | Media y desviación estándar (toda la distribución) | Solo el mínimo y el máximo (dos puntos) |
| Rango de salida | Sin límite fijo — media 0, desviación 1, pero los valores pueden ser negativos o mayores a 1 | Rango fijo, típicamente [0, 1] |
| Sensibilidad a outliers | Menor — un valor extremo pesa poco frente al promedio de todos los datos | Mayor — un solo valor extremo redefine el mínimo o el máximo, comprimiendo a todo el resto |

### Ejemplo numérico (5 clientes, `Monthly Charge` hipotético)

Valores: `[20, 40, 60, 80, 100]`. Media = 60, desviación estándar = 28.28. Mínimo = 20, máximo = 100.

| Valor original | Estandarizado `(x-60)/28.28` | Normalizado `(x-20)/80` |
|---|---|---|
| 20 | -1.414 | 0.00 |
| 40 | -0.707 | 0.25 |
| 60 | 0.000 | 0.50 |
| 80 | 0.707 | 0.75 |
| 100 | 1.414 | 1.00 |

Con datos "bien portados" (sin valores extremos) ambas dan una escala razonable, solo que en rangos distintos.

### Mismo ejemplo, agregando un outlier (`1000`)

Valores: `[20, 40, 60, 80, 100, 1000]`. Media = 216.67, desviación estándar = 351.27. Mínimo = 20, máximo = 1000.

| Valor original | Estandarizado | Normalizado |
|---|---|---|
| 20 | -0.560 | 0.0000 |
| 40 | -0.503 | 0.0204 |
| 60 | -0.446 | 0.0408 |
| 80 | -0.389 | 0.0612 |
| 100 | -0.332 | 0.0816 |
| 1000 | 2.230 | 1.0000 |

**Esto es lo que hay que poder explicar en la exposición:** con `MinMaxScaler`, los 5 valores originales (que antes ocupaban todo el rango de 0 a 1) quedan aplastados en una franja minúscula, de 0 a 0.08 — porque el outlier se convirtió en el nuevo "máximo" y estira toda la escala. Con `StandardScaler`, esos mismos 5 valores siguen relativamente distinguibles entre sí (de -0.56 a -0.33) porque la fórmula usa el promedio de **todos** los datos, no solo dos puntos extremos — un solo valor atípico pesa menos.

### Regla rápida para no confundir los nombres

Si el nombre de la función tiene **"Standard"** (`StandardScaler`) → estandarización, media 0 / desviación 1. Si tiene **"MinMax"** (`MinMaxScaler`) → normalización, rango fijo [0,1].

### Aplicación en este proyecto

- `eda_p1.py` (Random Forest / XGBoost / LightGBM): **ninguna de las dos** — los árboles dividen por umbral y son invariantes a la escala, así que ni estandarizar ni normalizar cambia el resultado del modelo (ver sección de `StandardScaler` más abajo para el detalle completo de por qué).
- `eda_p1_ml.py`, rama de regresión logística (modelo lineal, para probar separabilidad lineal): **estandarización** (`StandardScaler`), no normalización — es la opción más común para modelos lineales/basados en distancia porque es menos sensible a outliers que `MinMaxScaler`.

## Análisis de variables numéricas (eda_p1.py): outliers y multicolinealidad

Antes de armar la matriz final, `eda_p1.py` corre varios chequeos sobre las 12 numéricas, todos calculados **solo con train** (mismo criterio anti-leakage del resto del pipeline). Los hallazgos completos (tablas) quedan en `docs/hallazgos_numericas_p1.txt` cada vez que se corre el script; acá se documenta el razonamiento detrás de cada uno.

### 1. Varianza casi nula — ¿la variable distingue algo?

Mide qué proporción de las filas comparte el valor más frecuente (umbral 90%). Responde: *"¿esta variable aporta señal, o casi todos los clientes tienen el mismo valor?"* Solo `Total Refunds` lo supera (91.8% en 0). Es diagnóstico, no se elimina nada automáticamente.

**No es lo mismo que asimetría (skew).** Varianza casi nula mide si *hay* variación; skew mide la *forma* de esa variación (si tiene cola larga a un lado). Son preguntas independientes — prueba: `Total Charges` es muy asimétrica (skew 0.89) pero tiene varianza altísima (su valor más común es solo el 0.1% de los casos); `Tenure in Months` no es ni una cosa ni la otra (skew 0.15, valor más común solo 6%). Ver tabla comparativa completa en `docs/hallazgos_numericas_p1.txt`.

### 2. Descriptivos y sesgo (skew)

Media vs. mediana. `Total Refunds` (skew 4.08), `Total Extra Data Charges` (3.96) y `Number of Dependents` (2.12) tienen cola derecha fuerte. **No requiere ningún cambio para los árboles** (RF/XGBoost/LightGBM no asumen distribución simétrica, a diferencia de un modelo lineal). Sí es insumo directo para la Fase 5 del plan SHAP-Rule: la transformación arcsinh que ahí se aplica antes de calcular los cuartiles de los términos lingüísticos difusos se justifica exactamente por esta asimetría.

### 3. Outliers por IQR (regla de Tukey 1.5×IQR)

El % de "outliers" es alto en variables con muchos ceros (`Number of Dependents` 23.1%, `Total Extra Data Charges` 10.3%, `Total Refunds` 8.2%). **Caso a entender, no a corregir:** cuando ≥75% de los valores de una variable son 0, tanto Q1 como Q3 caen en 0, el IQR da 0, y los límites de Tukey colapsan a `[0, 0]`. Con ese rango, cualquier cliente que tenga un valor distinto de 0 —aunque sea perfectamente normal, como tener 1 dependiente— queda marcado como "outlier". No es un error de datos, es una limitación conocida del método IQR sobre variables concentradas en un solo valor (mismo patrón ya visto con `Number of Referrals` en el EDA original). `pct` es simplemente `n_outliers / filas_de_train × 100`. No se elimina ninguna fila automáticamente — el número avisa "revisa esta variable", la decisión de actuar o no es caso por caso.

**Para qué sirve encontrar outliers en general:** detectar errores de captura (como se hizo con `Monthly Charge` negativo), identificar casos extremos reales que conviene documentar (ej. clientes "premium" de `Total Revenue`), y —específico de este proyecto— alimentar los términos lingüísticos difusos de SHAP-Rule, que usan este mismo criterio de Tukey.

### 4. Matriz de correlación de Pearson

Pares con |r| ≥ 0.7: `Total Charges` ↔ `Total Revenue` (r=0.97), `Tenure in Months` ↔ `Total Revenue` (r=0.84), `Tenure in Months` ↔ `Total Charges` (r=0.82), `Total Long Distance Charges` ↔ `Total Revenue` (r=0.77). El heatmap se guarda como imagen en `docs/matriz_correlacion_pearson.png` (generado por `graficar_matriz_correlacion()` en `eda_p1.py`).

La correlación por pares solo detecta redundancia **entre dos variables**; no alcanza a ver cuando una variable es combinación de **varias** otras a la vez. Para eso se necesita VIF.

### 5. VIF (Variance Inflation Factor) y eliminación iterativa

**Qué mide:** para cada numérica, se hace una regresión lineal tratando de predecirla a partir de las demás numéricas, y `VIF = 1 / (1 − R²)`. Si esa regresión da un R² alto (la variable se puede "adivinar" casi perfectamente con las otras), el VIF se dispara. VIF > 10 es el corte estándar de multicolinealidad severa.

**Por qué es iterativo (3 pasos, no una sola pasada):** al quitar la variable más redundante, la relación entre las que quedan cambia, así que hay que recalcular VIF con el nuevo grupo cada vez — no basta con mirar la tabla original una sola vez.

- **Paso 1** (12 numéricas): 5 dieron VIF infinito **a la vez** — `Total Charges`, `Total Refunds`, `Total Extra Data Charges`, `Total Long Distance Charges`, `Total Revenue`. Causa raíz verificada numéricamente: existe una identidad contable exacta, `Total Revenue = Total Charges − Total Refunds + Total Long Distance Charges + Total Extra Data Charges` (no es una correlación aproximada, es una fórmula exacta). Como una es la suma/resta literal de las otras cuatro, cualquiera de las 5 se predice al 100% con las demás. Se elimina `Total Charges` (la primera del grupo empatado).
- **Paso 2** (11 numéricas): `Total Revenue` sale con VIF=16.7 — sigue siendo predecible casi por completo a partir de `Total Refunds` + `Total Long Distance Charges` + `Total Extra Data Charges`, que continúan presentes. Se elimina.
- **Paso 3** (10 numéricas): la más alta ahora es `Total Long Distance Charges` con VIF=5.35, ya por debajo de 10 → se detiene.

**Numéricas finales (10 de 12):** `Age`, `Number of Dependents`, `Number of Referrals`, `Tenure in Months`, `Avg Monthly Long Distance Charges`, `Avg Monthly GB Download`, `Monthly Charge`, `Total Refunds`, `Total Extra Data Charges`, `Total Long Distance Charges`. Eliminadas: `Total Charges`, `Total Revenue`.

**No se usa PCA** como método de eliminación porque combina las variables originales en componentes artificiales sin nombre — rompe la trazabilidad variable→regla que necesita SHAP-Rule (una regla debe poder decir *"SI Total Charges es alto..."*, no *"SI el componente 2 es alto..."*).

**Otros métodos posibles (además de VIF), para referencia:**
- *Número de condición / eigenvalues* de la matriz de correlación: mide multicolinealidad global del conjunto en un solo número, sin regresiones repetidas.
- *Clustering jerárquico de variables* (agrupar por `1 − |correlación|`, quedarse con una representante por grupo): más rápido que VIF iterativo, y la elección de qué variable representa al grupo queda a criterio de negocio en vez de depender del R² más alto.

Ningún método alternativo cambia el hallazgo de fondo (la identidad `Total Revenue = ...`) — solo son formas distintas de llegar a la misma conclusión.

**¿Por qué eliminar multicolineales si los árboles las toleran para predecir?** RF/XGBoost/LightGBM no pierden precisión con variables correlacionadas. La razón para igual eliminarlas es la **calidad de las explicaciones SHAP**: cuando dos variables están muy correlacionadas, el modelo reparte la contribución entre ambas de forma inconsistente entre árboles del ensamble — dos instancias parecidas pueden salir con reglas distintas solo por cómo se repartió el peso. Eso afecta directamente la métrica de consistencia (Levenshtein) que el plan SHAP-Rule mide en la Fase 6.

La matriz final que guarda `eda_p1.py` en `data/processed/` combina estas 10 numéricas (sin escalar) con las 18 categóricas ordinales: `X_train_prep` queda en 28 columnas (antes eran 30 con las 12 numéricas completas).
