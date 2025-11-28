# Ponia

Query pandas DataFrames with natural language using OpenAI function calling.

## Installation

```bash
pip install -e .
```

## Usage

```python
import pandas as pd
from ponia import ponia

# Set your API key as environment variable
# export OPENAI_API_KEY="your-api-key"

df = pd.DataFrame({
    'product': ['apple', 'banana', 'orange'],
    'monday': [10, 15, 8],
    'tuesday': [12, 18, 9],
    'wednesday': [8, 20, 11]
})

# Query in natural language
answer = ponia(df, "Which column has the largest value?")
print(answer)
# "The column 'wednesday' has the largest value, which is 20..."

answer = ponia(df, "What is the average of monday sales?")
print(answer)
# "The average of column 'monday' is 11.0"

# Get raw JSON result instead of natural language
result = ponia(df, "What is the max value?", raw=True)
print(result)
# {'value': 20, 'column': 'wednesday', 'row': 1}
```

## Features

- **Zero data leakage**: Data is never sent to OpenAI, only the question
- **No eval()**: Does not execute AI-generated code, only predefined functions
- **Minimal**: Minimal dependencies (pandas, openai)
- **Extensible**: Easy to add new functions

## Available Functions

- `find_max_value_location` - Global max value and location
- `find_min_value_location` - Global min value and location
- `get_column_max/min/sum/mean/median/std` - Column statistics
- `count_rows/columns/unique` - Counts
- `filter_by_value/comparison` - Row filtering
- `filter_and_aggregate` - Filter then aggregate (e.g., "average salary for engineers")
- `group_aggregate` - Group by and aggregate
- `get_correlation` - Correlation between columns
- `get_top_n_rows/bottom_n_rows` - Top/bottom N rows
- And more...

## Tests

```bash
pip install -e ".[dev]"
pytest
```
