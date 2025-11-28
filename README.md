# Ponia

Query pandas DataFrames with natural language using OpenAI function calling.

## Instalación

```bash
pip install -e .
```

## Uso

```python
import pandas as pd
from ponia import ponia

# Configura tu API key como variable de entorno
# export OPENAI_API_KEY="tu-api-key"

df = pd.DataFrame({
    'producto': ['manzana', 'banana', 'naranja'],
    'lunes': [10, 15, 8],
    'martes': [12, 18, 9],
    'miercoles': [8, 20, 11]
})

# Hacer consultas en lenguaje natural
respuesta = ponia(df, "¿Qué columna tiene el valor más grande?")
print(respuesta)
# "La columna 'miercoles' tiene el valor más grande, el cual es 20..."

respuesta = ponia(df, "¿Cuál es el promedio de ventas del lunes?")
print(respuesta)
# "El promedio de la columna 'lunes' es 11.0"
```

## Características

- **Zero data leakage**: Los datos nunca se envían a OpenAI, solo la pregunta
- **Sin eval()**: No ejecuta código generado por IA, solo funciones predefinidas
- **Minimal**: Dependencias mínimas (pandas, openai)
- **Extensible**: Fácil de agregar nuevas funciones

## Funciones disponibles

- `find_max_value_location` - Valor máximo global y ubicación
- `find_min_value_location` - Valor mínimo global y ubicación
- `get_column_max/min/sum/mean/median/std` - Estadísticas por columna
- `count_rows/columns/unique` - Conteos
- `filter_by_value/comparison` - Filtrado de filas
- `group_aggregate` - Agrupación y agregación
- `get_correlation` - Correlación entre columnas
- `get_top_n_rows/bottom_n_rows` - Top/bottom N filas
- Y más...

## Tests

```bash
pip install -e ".[dev]"
pytest
```
