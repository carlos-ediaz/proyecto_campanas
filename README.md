# Clasificación de campañas de marketing: ¿ROI alto o bajo?

Dashboard interactivo hecho con **Dash + Plotly** que explora campañas de marketing digital y evalúa modelos de **clasificación binaria** para predecir si una campaña tendrá un **ROI alto** o un **ROI bajo**.

> **Sobre los datos.** El proyecto usa un dataset público de campañas de marketing (~200,000 filas). Fuente: _[pega aquí el enlace de la página del dataset]_ · Licencia: _[indica la licencia que figura en esa página]_.
> Las distribuciones de varias columnas (ROI entre 2 y 8, tasa de conversión entre 0.01 y 0.15, identificadores uniformes) son casi planas, lo cual es típico de **datos generados artificialmente**. Si la página del dataset confirma que es sintético, indícalo aquí. En cualquier caso, el proyecto es de **uso académico** y sus resultados no deben usarse para decisiones reales de negocio.

## Estructura del proyecto

```
proyecto_campanas/
├── app.py                 # Aplicación completa (datos, modelos, interfaz, callbacks, servidor)
├── requirements.txt       # Dependencias
├── assets/
│   └── style.css          # Estilos (Dash lo carga automáticamente)
└── data/
    └── marketing_campaign_dataset.csv
```

`app.py` sigue un flujo lineal: importar librerías → cargar los datos en memoria → crear la aplicación y definir la interfaz → registrar los callbacks → iniciar el servidor.

## Cómo ejecutarlo

1. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

2. Coloca el CSV en `data/marketing_campaign_dataset.csv`. Si tiene otro nombre, la app usa el primer `.csv` que encuentre en `data/`, o puedes indicar la ruta con la variable de entorno `DATA_PATH`.

3. Desde la carpeta del proyecto ejecuta:

   ```bash
   python app.py
   ```

4. Abre <http://127.0.0.1:8050> en el navegador.

Al arrancar, la app entrena los modelos en memoria y tarda alrededor de 30 segundos a un minuto antes de mostrar que el servidor está corriendo.

**Muestra de datos.** El dataset completo tiene ~200,000 filas. Para que el entrenamiento sea rápido, la app trabaja con una **muestra aleatoria de 20,000 campañas** (semilla fija). Puedes cambiarlo con la variable `MAX_FILAS` (por ejemplo, `MAX_FILAS=50000`); el umbral de ROI alto siempre se calcula con todas las filas.

## Datos

Cada fila es una campaña. Columnas del CSV:

| Columna original | Descripción | Uso en el proyecto |
|---|---|---|
| `Campaign_ID` | Identificador de la campaña | Solo identificación |
| `Company` | Empresa (p. ej. TechCorp, Alpha Innovations) | Predictora |
| `Campaign_Type` | Email, Influencer, Display, Search, Social Media | Predictora |
| `Target_Audience` | Audiencia objetivo (p. ej. Men 18-24, Women 35-44, All Ages) | Predictora |
| `Duration` | Duración (15, 30, 45 o 60 días; se convierte a número) | Predictora |
| `Channel_Used` | Canal (Google Ads, YouTube, Instagram, Facebook, Website, Email) | Predictora |
| `Conversion_Rate` | Tasa de conversión | **Excluida** (ver abajo); solo se muestra |
| `Acquisition_Cost` | Costo de adquisición en dólares (se limpia el formato `$16,174.00`) | Predictora |
| `ROI` | Retorno de la inversión | **Origen de la variable objetivo** |
| `Location` | Ubicación (Miami, New York, Chicago, Los Angeles, Houston) | Predictora |

## Variable objetivo

Se crea la variable binaria `roi_alto`:

- **1 (Alto):** `ROI` es mayor o igual a la **mediana** del ROI.
- **0 (Bajo):** `ROI` es menor que la mediana.

Se usa la mediana porque deja las dos clases balanceadas (aprox. 50 % y 50 %), lo que facilita interpretar métricas como la exactitud.

## Variables usadas en el modelo

**Numéricas** (se estandarizan con `StandardScaler`): `costo_adquisicion`, `duracion_dias`.

**Categóricas** (se codifican con `OneHotEncoder`): `empresa`, `tipo_campana`, `audiencia`, `canal`, `ubicacion`.

**Variables excluidas a propósito:**

- `ROI`: es la variable de la que se deriva el objetivo (fuga de información).
- `Conversion_Rate`: es un **resultado** de la campaña, igual que el ROI, y no se conoce al planearla. Incluirla haría que el modelo usara información que no tendría al decidir una campaña. Puedes añadirla a la lista `NUM` de `app.py` si quieres comprobar el efecto.
- `Campaign_ID`: es solo un identificador.

## Modelos y por qué se eligieron

Se comparan tres iteraciones, de la más simple a la más flexible:

| Modelo | Qué usa | Qué pregunta responde |
|---|---|---|
| **v1 Regresión logística (costo)** | Solo `costo_adquisicion` | ¿Basta con el costo para anticipar un ROI alto? (línea base) |
| **v2 Regresión logística múltiple** | Las 7 variables | Con la información completa de planeación, ¿mejora la predicción usando relaciones lineales? |
| **v3 Random Forest** | Las 7 variables | ¿Hay combinaciones de variables (por ejemplo, canal × audiencia) que un modelo lineal no ve? |

- **Clasificación binaria:** plantear "ROI alto o bajo" es más directo para decidir dónde invertir que predecir el valor exacto del ROI.
- **Regresión logística:** es interpretable, estable y sirve como punto de comparación.
- **Random Forest:** captura relaciones no lineales e interacciones. Es el modelo seleccionado por defecto en el dashboard, pero **no se asume que sea el mejor**: la pestaña de rendimiento permite compararlo con los otros dos.

## Evaluación

- **Validación cruzada estratificada de 5 pliegues:** cada campaña se clasifica con un modelo que no la vio al entrenar, así las métricas no se inflan y siguen siendo válidas al aplicar filtros.
- **Umbral de decisión:** 0.5 sobre la probabilidad de ROI alto.
- **Métricas:** exactitud (*accuracy*), precisión, sensibilidad (*recall*), F1 y ROC-AUC.
- **Importancia de variables:** por permutación del Random Forest (ajustado con 80 % de la muestra y evaluado con el 20 % restante), medida como caída del ROC-AUC.

**Cómo leer el ROC-AUC:** 0.50 equivale a adivinar al azar; cuanto más cerca de 1.0, mejor separa el modelo los ROI altos de los bajos.

### Resultado esperado con estos datos

En pruebas con datos de estructura idéntica pero con variables sin relación con el ROI, los tres modelos obtienen un **ROC-AUC ≈ 0.50** (no mejor que el azar), y la importancia de las variables es cercana a cero. Si al ejecutar el dataset real obtienes algo similar, no es un error de la aplicación: significa que las variables disponibles no permiten predecir el ROI. Es un hallazgo válido y conviene reportarlo así.

_[Añade aquí los valores reales de ROC-AUC de tus tres modelos tras ejecutar la app.]_

## Qué muestra el dashboard

- **Filtros:** canal, tipo de campaña, ubicación, audiencia, duración, rango de costo de adquisición y modelo a evaluar.
- **KPIs:** número de campañas, ROI medio, costo de adquisición medio, tasa de conversión media y porcentaje de campañas con ROI alto.
- **Panorama:** ROI medio por canal, porcentaje de ROI alto por tipo de campaña, dispersión costo-ROI y distribución del ROI con el umbral.
- **Rendimiento del modelo:** métricas, matriz de confusión, curva ROC, comparación de modelos por ROC-AUC, importancia de variables y una lectura ejecutiva.
- **Errores de clasificación:** las campañas que el modelo seleccionado clasifica peor.

## Limitaciones

- Los resultados dependen de una muestra de 20,000 campañas y de la mediana como umbral, que es una decisión de diseño.
- Los factores muestran **asociación predictiva, no causalidad**.
- No hay ajuste de hiperparámetros: los modelos usan valores razonables fijos.
- El dataset no incluye fechas, por lo que no hay análisis temporal.

## Tecnologías

Python · Dash · Plotly · pandas · NumPy · scikit-learn

## Licencia y uso

Proyecto con fines académicos. Antes de publicarlo, respeta la licencia del dataset original y añade el archivo de licencia que prefieras para el código (por ejemplo, MIT).
