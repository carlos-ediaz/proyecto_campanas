# Predicción de abandono de clientes (churn) en telecomunicaciones

Dashboard interactivo hecho con **Dash + Plotly** que explora el abandono de clientes de una empresa de telecomunicaciones y evalúa modelos de **clasificación binaria** para predecir si un cliente **abandona** o **se queda**.

**Autores:** _[Nombre 1]_ y _[Nombre 2]_ · Programación para Ciencia de Datos · Universidad Compensar (tercer parcial)

> **Sobre los datos.** El proyecto usa el dataset público **Telco Customer Churn** (7.043 clientes, 21 columnas). Fuente: _[pega aquí el enlace de la página del dataset en Kaggle]_ · Licencia: _[indica la licencia que figura en esa página]_.
> Es un conjunto de datos de ejemplo de una sola empresa, así que el proyecto es de **uso académico** y sus resultados no deben usarse para decisiones reales de negocio.

## Problema y objetivo

Conseguir un cliente nuevo cuesta más que conservar uno actual. El proyecto busca:

1. Identificar qué características del cliente y del servicio se asocian con el abandono.
2. Construir un modelo de clasificación binaria que estime el riesgo de abandono de cada cliente, para priorizar acciones de retención.

## Estructura del proyecto

```
proyecto_churn/
├── app.py                 # Aplicación completa (datos, limpieza, modelos, interfaz, callbacks, servidor)
├── requirements.txt       # Dependencias
├── README.md
├── assets/
│   └── style.css          # Estilos (Dash lo carga automáticamente)
└── data/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

`app.py` sigue un flujo lineal: importar librerías → cargar y limpiar los datos → entrenar y evaluar los modelos → crear la interfaz → registrar los callbacks → iniciar el servidor.

## Cómo ejecutarlo

1. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Coloca el CSV en `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. Si tiene otro nombre, la app usa el primer `.csv` que encuentre en `data/`, o puedes indicar la ruta con la variable de entorno `DATA_PATH`.

3. Desde la carpeta del proyecto ejecuta:

   ```bash
   python app.py
   ```

4. Abre <http://127.0.0.1:8050> en el navegador.

Al arrancar, la app limpia los datos y entrena los modelos en memoria, lo que tarda unos 15 a 30 segundos antes de mostrar que el servidor está corriendo.

## Datos

Cada fila es un cliente. Se usan 15 de las 21 columnas del CSV:

| Columna original | Nombre en el proyecto | Descripción | Uso |
|---|---|---|---|
| `customerID` | `id_cliente` | Identificador del cliente | Solo identificación |
| `gender` | `genero` | Género | Predictora |
| `SeniorCitizen` | `adulto_mayor` | Si es adulto mayor (0/1 → No/Sí) | Predictora |
| `Partner` | `pareja` | Si tiene pareja | Predictora |
| `Dependents` | `dependientes` | Si tiene dependientes | Predictora |
| `tenure` | `antiguedad_meses` | Meses como cliente (0 a 72) | Predictora |
| `InternetService` | `internet` | DSL, fibra óptica o sin internet | Predictora |
| `OnlineSecurity` | `seguridad_online` | Servicio de seguridad online | Predictora |
| `TechSupport` | `soporte_tecnico` | Servicio de soporte técnico | Predictora |
| `Contract` | `contrato` | Mensual, 1 año o 2 años | Predictora |
| `PaperlessBilling` | `factura_digital` | Si recibe factura digital | Predictora |
| `PaymentMethod` | `metodo_pago` | Cheque electrónico, cheque por correo, transferencia o tarjeta automática | Predictora |
| `MonthlyCharges` | `cargo_mensual` | Cargo mensual en dólares | Predictora |
| `TotalCharges` | `cargo_total` | Cargo total acumulado en dólares | Predictora |
| `Churn` | `churn` | Si el cliente abandonó (Yes/No) | **Variable objetivo** |

No se usan `PhoneService`, `MultipleLines`, `OnlineBackup`, `DeviceProtection`, `StreamingTV` ni `StreamingMovies`, para mantener el modelo y el dashboard más simples.

## Limpieza y preparación de datos

Cada paso queda registrado en la pestaña **Contexto y datos** del dashboard.

| Paso | Qué se encontró / qué se hizo |
|---|---|
| Duplicados | 0 registros duplicados (por fila o por `id_cliente`) |
| `TotalCharges` como texto | La columna viene como texto y se convierte a número |
| Valores en blanco en `TotalCharges` | 11 registros, todos con antigüedad 0 (clientes nuevos aún sin facturar). Se imputan con antigüedad × cargo mensual (= 0) en lugar de eliminarlos |
| Nulos y rangos | 0 nulos restantes; se verifica antigüedad entre 0 y 72 meses y cargos positivos |
| Transformaciones | `Churn` Yes/No → 1/0; categorías traducidas al español; `SeniorCitizen` 0/1 → No/Sí |
| Tramos de antigüedad | 0-12, 13-24, 25-48 y 49-72 meses (solo para el análisis visual, no entra al modelo) |

Resultado: **7.043 clientes** listos para analizar.

## Variable objetivo

`churn` es binaria: **1** si el cliente abandonó y **0** si se quedó. Abandonó el **26,5 %**, así que las clases están **desbalanceadas**. Por eso los modelos se entrenan con `class_weight` balanceado y la evaluación se apoya en recall, F1 y ROC-AUC, además de la exactitud.

## Variables usadas en el modelo

**Numéricas** (se estandarizan con `StandardScaler`): `antiguedad_meses`, `cargo_mensual`, `cargo_total`.

**Categóricas** (se codifican con `OneHotEncoder`): `genero`, `adulto_mayor`, `pareja`, `dependientes`, `internet`, `seguridad_online`, `soporte_tecnico`, `contrato`, `factura_digital`, `metodo_pago`.

**Excluidas:** `id_cliente` (solo identifica) y `churn` (es la variable que se predice).

## Modelos y por qué se eligieron

Se comparan tres iteraciones, de la más simple a la más flexible:

| Modelo | Qué usa | Qué pregunta responde |
|---|---|---|
| **v1 Regresión logística (antigüedad)** | Solo `antiguedad_meses` | ¿Basta con la antigüedad para anticipar el abandono? (línea base) |
| **v2 Regresión logística múltiple** | Las 13 variables | Con toda la información del cliente, ¿mejora la predicción con relaciones lineales? |
| **v3 Random Forest** | Las 13 variables | ¿Hay combinaciones de variables que un modelo lineal no ve? |

- **Clasificación binaria:** responde directamente a la decisión de negocio (¿actuar sobre este cliente o no?).
- **Regresión logística:** es interpretable, estable y sirve como punto de comparación.
- **Random Forest** (300 árboles, `min_samples_leaf=10`): captura relaciones no lineales e interacciones. El dashboard selecciona por defecto el modelo con **mayor ROC-AUC**, y la pestaña de rendimiento permite compararlo con los otros dos.

## Evaluación

- **Validación cruzada estratificada de 5 pliegues:** cada cliente se clasifica con un modelo que no lo vio al entrenar, así las métricas no se inflan y siguen siendo válidas al aplicar filtros.
- **Umbral de decisión:** 0,5 sobre la probabilidad de abandono.
- **Métricas:** exactitud (*accuracy*), precisión, sensibilidad (*recall*), F1 y ROC-AUC.
- **Importancia de variables:** por permutación del modelo seleccionado (ajustado con 80 % de los datos y evaluado con el 20 % restante), medida como caída del ROC-AUC.

**Cómo leer el ROC-AUC:** 0,50 equivale a adivinar al azar; cuanto más cerca de 1,0, mejor separa el modelo a quienes abandonan de quienes se quedan.

### Resultados (todos los clientes, validación cruzada)

| Modelo | Exactitud | Precisión | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| v1 Regresión logística (antigüedad) | 64,1 % | 40,4 % | 73,7 % | 0,522 | 0,739 |
| v2 Regresión logística múltiple | 74,1 % | 50,7 % | 80,3 % | 0,622 | 0,842 |
| **v3 Random Forest** | **76,7 %** | **54,5 %** | 75,0 % | **0,631** | **0,843** |

Matriz de confusión del Random Forest (filas = clase real, columnas = clase predicha):

| | Predice «se queda» | Predice «abandona» |
|---|---|---|
| **Real «se queda»** | 4.002 | 1.172 |
| **Real «abandona»** | 468 | 1.401 |

El modelo detecta 3 de cada 4 clientes que realmente abandonan. Las variables más importantes son `contrato`, `antiguedad_meses` e `internet`.

## Principales hallazgos

| Factor | Abandono |
|---|---|
| Contrato mensual / 1 año / 2 años | 42,7 % / 11,3 % / 2,8 % |
| Antigüedad 0-12 meses / 49-72 meses | 47,4 % / 9,5 % |
| Pago con cheque electrónico | 45,3 % (frente a 15-19 % con las demás formas de pago) |
| Fibra óptica / DSL | 41,9 % / 19,0 % |
| Sin soporte técnico / con soporte técnico | 41,6 % / 15,2 % |
| Contrato mensual + fibra óptica + cheque electrónico | 60,4 % (1.307 clientes) |

Los clientes que abandonaron representaban unos **139 mil dólares de ingreso mensual**.

**Recomendaciones:** incentivar la migración a contratos largos, promover el pago automático, reforzar el acompañamiento y el soporte técnico durante los primeros meses y revisar el precio y la calidad de la fibra óptica.

## Qué muestra el dashboard

El dashboard se organiza en cinco pestañas:

- **Contexto y datos:** problema de negocio, objetivo, datos y tabla con la limpieza realizada.
- **Panorama:** % de abandono por contrato, por método de pago y por tramo de antigüedad, distribución del cargo mensual y un cuadro de hallazgos y recomendaciones que se recalcula con los filtros.
- **Rendimiento del modelo:** métricas, matriz de confusión, curva ROC, comparación de modelos por ROC-AUC, importancia de variables y una lectura ejecutiva.
- **Resultados predictivos:** clientes activos con mayor riesgo de abandono (para priorizar retención) y los clientes que el modelo clasifica peor.
- **Simulador:** se elige el perfil de un cliente y se muestra su probabilidad de abandono en un medidor con riesgo bajo, medio o alto.

**Filtros** (barra lateral): contrato, servicio de internet, método de pago, adulto mayor, rango de antigüedad y modelo a evaluar. **Indicadores:** número de clientes, tasa de abandono, cargo mensual medio, antigüedad media e ingreso mensual en riesgo.

## Limitaciones

- Los factores muestran **asociación predictiva, no causalidad**.
- Los modelos se entrenan con clases balanceadas, así que las probabilidades tienden a **sobrestimar el riesgo**; el simulador sirve sobre todo para comparar perfiles entre sí.
- El umbral de decisión es 0,5 y no hay ajuste de hiperparámetros: los modelos usan valores razonables fijos.
- Es un conjunto de datos de una sola empresa y de un solo momento, sin fechas, por lo que no hay análisis temporal.
- El simulador usa el modelo entrenado con todos los datos, mientras que las métricas del dashboard salen de validación cruzada.

## Tecnologías

Python · Dash · Plotly · pandas · NumPy · scikit-learn

## Licencia y uso

Proyecto con fines académicos. Antes de publicarlo, respeta la licencia del dataset original y añade el archivo de licencia que prefieras para el código (por ejemplo, MIT).
