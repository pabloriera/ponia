"""Catalog of functions for processing DataFrames.

Each function receives (df, params: dict) and returns a serializable dict.
These functions are called via OpenAI function calling.
"""

import pandas as pd
from typing import Any


def find_max_value_location(df: pd.DataFrame, params: dict) -> dict:
    """Find the global maximum value in the DataFrame and its location.

    Scans all numeric columns to find the single largest value and returns
    both the value and its exact position (column and row).

    Args:
        df: The pandas DataFrame to analyze.
        params: Empty dict (no parameters required).

    Returns:
        dict: Contains 'value' (the max value), 'column' (column name),
              and 'row' (row index). Returns {'error': str} if no numeric columns.

    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> find_max_value_location(df, {})
        {'value': 4, 'column': 'b', 'row': 1}
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return {"error": "No numeric columns found"}

    max_val = numeric_df.max().max()
    col = numeric_df.max().idxmax()
    row = numeric_df[col].idxmax()
    return {"value": max_val, "column": col, "row": row}


def find_min_value_location(df: pd.DataFrame, params: dict) -> dict:
    """Find the global minimum value in the DataFrame and its location.

    Scans all numeric columns to find the single smallest value and returns
    both the value and its exact position (column and row).

    Args:
        df: The pandas DataFrame to analyze.
        params: Empty dict (no parameters required).

    Returns:
        dict: Contains 'value' (the min value), 'column' (column name),
              and 'row' (row index). Returns {'error': str} if no numeric columns.

    Example:
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> find_min_value_location(df, {})
        {'value': 1, 'column': 'a', 'row': 0}
    """
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return {"error": "No numeric columns found"}

    min_val = numeric_df.min().min()
    col = numeric_df.min().idxmin()
    row = numeric_df[col].idxmin()
    return {"value": min_val, "column": col, "row": row}


def get_column_max(df: pd.DataFrame, params: dict) -> dict:
    """Get the maximum value of a specific column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) - the column name to analyze.

    Returns:
        dict: Contains 'column', 'max_value', and 'row' (index of max).
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'price': [10, 25, 15]})
        >>> get_column_max(df, {'column': 'price'})
        {'column': 'price', 'max_value': 25, 'row': 1}
    """
    column = params.get("column")
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    max_val = df[column].max()
    max_idx = df[column].idxmax()
    return {"column": column, "max_value": max_val, "row": max_idx}


def get_column_min(df: pd.DataFrame, params: dict) -> dict:
    """Get the minimum value of a specific column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) - the column name to analyze.

    Returns:
        dict: Contains 'column', 'min_value', and 'row' (index of min).
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'price': [10, 25, 15]})
        >>> get_column_min(df, {'column': 'price'})
        {'column': 'price', 'min_value': 10, 'row': 0}
    """
    column = params.get("column")
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    min_val = df[column].min()
    min_idx = df[column].idxmin()
    return {"column": column, "min_value": min_val, "row": min_idx}


def get_column_sum(df: pd.DataFrame, params: dict) -> dict:
    """Calculate the sum of all values in a column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) - the column name to sum.

    Returns:
        dict: Contains 'column' and 'sum'.
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'sales': [100, 200, 150]})
        >>> get_column_sum(df, {'column': 'sales'})
        {'column': 'sales', 'sum': 450}
    """
    column = params.get("column")
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    total = df[column].sum()
    return {"column": column, "sum": total}


def get_column_mean(df: pd.DataFrame, params: dict) -> dict:
    """Calculate the arithmetic mean of a column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) - the column name.

    Returns:
        dict: Contains 'column' and 'mean'.
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'score': [80, 90, 85]})
        >>> get_column_mean(df, {'column': 'score'})
        {'column': 'score', 'mean': 85.0}
    """
    column = params.get("column")
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    mean_val = df[column].mean()
    return {"column": column, "mean": mean_val}


def get_column_median(df: pd.DataFrame, params: dict) -> dict:
    """Calculate the median of a column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) - the column name.

    Returns:
        dict: Contains 'column' and 'median'.
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'value': [1, 3, 5, 7, 9]})
        >>> get_column_median(df, {'column': 'value'})
        {'column': 'value', 'median': 5.0}
    """
    column = params.get("column")
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    median_val = df[column].median()
    return {"column": column, "median": median_val}


def get_column_std(df: pd.DataFrame, params: dict) -> dict:
    """Calculate the standard deviation of a column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) - the column name.

    Returns:
        dict: Contains 'column' and 'std'.
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'value': [2, 4, 4, 4, 5, 5, 7, 9]})
        >>> get_column_std(df, {'column': 'value'})
        {'column': 'value', 'std': 2.0}
    """
    column = params.get("column")
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    std_val = df[column].std()
    return {"column": column, "std": std_val}


def count_rows(df: pd.DataFrame, params: dict) -> dict:
    """Count the total number of rows in the DataFrame.

    Args:
        df: The pandas DataFrame to analyze.
        params: Empty dict (no parameters required).

    Returns:
        dict: Contains 'total_rows' with the row count.

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3]})
        >>> count_rows(df, {})
        {'total_rows': 3}
    """
    return {"total_rows": len(df)}


def count_columns(df: pd.DataFrame, params: dict) -> dict:
    """Count the total number of columns in the DataFrame.

    Args:
        df: The pandas DataFrame to analyze.
        params: Empty dict (no parameters required).

    Returns:
        dict: Contains 'total_columns' with the column count.

    Example:
        >>> df = pd.DataFrame({'a': [1], 'b': [2], 'c': [3]})
        >>> count_columns(df, {})
        {'total_columns': 3}
    """
    return {"total_columns": len(df.columns)}


def count_unique(df: pd.DataFrame, params: dict) -> dict:
    """Count unique values in a column and show sample values.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) - the column name.

    Returns:
        dict: Contains 'column', 'unique_count', and 'sample_values' (up to 20).
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'color': ['red', 'blue', 'red', 'green']})
        >>> count_unique(df, {'column': 'color'})
        {'column': 'color', 'unique_count': 3, 'sample_values': ['red', 'blue', 'green']}
    """
    column = params.get("column")
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    unique_count = df[column].nunique()
    unique_values = df[column].unique().tolist()[:20]  # Limit to 20
    return {"column": column, "unique_count": unique_count, "sample_values": unique_values}


def get_value_counts(df: pd.DataFrame, params: dict) -> dict:
    """Get frequency count for each value in a column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) - the column name.
                Optional 'top_n' (int) - number of top values (default: 10).

    Returns:
        dict: Contains 'column' and 'value_counts' dict.
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'fruit': ['apple', 'banana', 'apple', 'apple']})
        >>> get_value_counts(df, {'column': 'fruit'})
        {'column': 'fruit', 'value_counts': {'apple': 3, 'banana': 1}}
    """
    column = params.get("column")
    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    top_n = params.get("top_n", 10)
    counts = df[column].value_counts().head(top_n).to_dict()
    return {"column": column, "value_counts": counts}


def filter_by_value(df: pd.DataFrame, params: dict) -> dict:
    """Filter rows where a column equals a specific value.

    Args:
        df: The pandas DataFrame to filter.
        params: Dict with 'column' (str) and 'value' (any) - the filter criteria.

    Returns:
        dict: Contains 'column', 'filter_value', 'matching_rows' count,
              and 'sample' (first 5 matching rows as records).
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'city': ['NY', 'LA', 'NY'], 'pop': [8, 4, 8]})
        >>> filter_by_value(df, {'column': 'city', 'value': 'NY'})
        {'column': 'city', 'filter_value': 'NY', 'matching_rows': 2, 'sample': [...]}
    """
    column = params.get("column")
    value = params.get("value")

    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    filtered = df[df[column] == value]
    return {
        "column": column,
        "filter_value": value,
        "matching_rows": len(filtered),
        "sample": filtered.head(5).to_dict(orient="records")
    }


def filter_by_comparison(df: pd.DataFrame, params: dict) -> dict:
    """Filter rows by numeric comparison (>, <, >=, <=).

    Args:
        df: The pandas DataFrame to filter.
        params: Dict with 'column' (str), 'operator' (str: 'gt', 'lt', 'gte', 'lte'),
                and 'value' (number) - the comparison criteria.

    Returns:
        dict: Contains 'column', 'operator', 'value', 'matching_rows' count,
              and 'sample' (first 5 matching rows).
              Returns {'error': str} if column not found or invalid operator.

    Example:
        >>> df = pd.DataFrame({'age': [25, 30, 35, 40]})
        >>> filter_by_comparison(df, {'column': 'age', 'operator': 'gte', 'value': 35})
        {'column': 'age', 'operator': 'gte', 'value': 35, 'matching_rows': 2, 'sample': [...]}
    """
    column = params.get("column")
    operator = params.get("operator")  # gt, lt, gte, lte
    value = params.get("value")

    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    ops = {
        "gt": df[column] > value,
        "lt": df[column] < value,
        "gte": df[column] >= value,
        "lte": df[column] <= value,
    }

    if operator not in ops:
        return {"error": f"Invalid operator '{operator}'. Use: gt, lt, gte, lte"}

    filtered = df[ops[operator]]
    return {
        "column": column,
        "operator": operator,
        "value": value,
        "matching_rows": len(filtered),
        "sample": filtered.head(5).to_dict(orient="records")
    }


def group_aggregate(df: pd.DataFrame, params: dict) -> dict:
    """Group by a column and apply an aggregation function.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'group_by' (str) - column to group by,
                'agg_column' (str, optional) - column to aggregate,
                'agg_func' (str) - one of 'sum', 'mean', 'count', 'min', 'max'.

    Returns:
        dict: Contains 'group_by', 'agg_column', 'agg_func', and 'result' dict.
              Returns {'error': str} if columns not found.

    Example:
        >>> df = pd.DataFrame({'dept': ['A', 'A', 'B'], 'salary': [50, 60, 70]})
        >>> group_aggregate(df, {'group_by': 'dept', 'agg_column': 'salary', 'agg_func': 'mean'})
        {'group_by': 'dept', 'agg_column': 'salary', 'agg_func': 'mean', 'result': {'A': 55.0, 'B': 70.0}}
    """
    group_by = params.get("group_by")
    agg_column = params.get("agg_column")
    agg_func = params.get("agg_func", "mean")  # sum, mean, count, min, max

    if group_by not in df.columns:
        return {"error": f"Column '{group_by}' not found"}
    if agg_column and agg_column not in df.columns:
        return {"error": f"Column '{agg_column}' not found"}

    if agg_func == "count":
        result = df.groupby(group_by, observed=True).size().to_dict()
    else:
        result = df.groupby(group_by, observed=True)[agg_column].agg(agg_func).to_dict()

    return {
        "group_by": group_by,
        "agg_column": agg_column,
        "agg_func": agg_func,
        "result": result
    }


def compare_columns(df: pd.DataFrame, params: dict) -> dict:
    """Compare basic statistics between two columns.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column1' (str) and 'column2' (str) - columns to compare.

    Returns:
        dict: Contains 'column1' and 'column2' dicts, each with 'name', 'mean', 'sum'.
              Returns {'error': str} if columns not found.

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        >>> compare_columns(df, {'column1': 'a', 'column2': 'b'})
        {'column1': {'name': 'a', 'mean': 2.0, 'sum': 6}, 'column2': {'name': 'b', 'mean': 5.0, 'sum': 15}}
    """
    col1 = params.get("column1")
    col2 = params.get("column2")

    if col1 not in df.columns:
        return {"error": f"Column '{col1}' not found"}
    if col2 not in df.columns:
        return {"error": f"Column '{col2}' not found"}

    return {
        "column1": {
            "name": col1,
            "mean": df[col1].mean() if pd.api.types.is_numeric_dtype(df[col1]) else None,
            "sum": df[col1].sum() if pd.api.types.is_numeric_dtype(df[col1]) else None,
        },
        "column2": {
            "name": col2,
            "mean": df[col2].mean() if pd.api.types.is_numeric_dtype(df[col2]) else None,
            "sum": df[col2].sum() if pd.api.types.is_numeric_dtype(df[col2]) else None,
        }
    }


def get_column_names(df: pd.DataFrame, params: dict) -> dict:
    """List all column names and their data types.

    Args:
        df: The pandas DataFrame to analyze.
        params: Empty dict (no parameters required).

    Returns:
        dict: Contains 'columns' (list of names) and 'dtypes' (dict of column: dtype).

    Example:
        >>> df = pd.DataFrame({'name': ['Alice'], 'age': [30]})
        >>> get_column_names(df, {})
        {'columns': ['name', 'age'], 'dtypes': {'name': 'object', 'age': 'int64'}}
    """
    columns = df.columns.tolist()
    dtypes = {col: str(df[col].dtype) for col in columns}
    return {"columns": columns, "dtypes": dtypes}


def describe_dataframe(df: pd.DataFrame, params: dict) -> dict:
    """Get descriptive statistics for the DataFrame.

    Provides count, mean, std, min, max, and percentiles for numeric columns,
    plus count, unique, top, freq for categorical columns.

    Args:
        df: The pandas DataFrame to analyze.
        params: Empty dict (no parameters required).

    Returns:
        dict: Contains 'shape' (rows/columns) and 'statistics' (describe output).

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'x']})
        >>> result = describe_dataframe(df, {})
        >>> result['shape']
        {'rows': 3, 'columns': 2}
    """
    desc = df.describe(include='all').to_dict()
    return {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "statistics": desc
    }


def get_null_counts(df: pd.DataFrame, params: dict) -> dict:
    """Count null/missing values per column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Empty dict (no parameters required).

    Returns:
        dict: Contains 'null_counts' (dict per column) and 'total_nulls' (int).

    Example:
        >>> df = pd.DataFrame({'a': [1, None, 3], 'b': [None, None, 3]})
        >>> get_null_counts(df, {})
        {'null_counts': {'a': 1, 'b': 2}, 'total_nulls': 3}
    """
    null_counts = df.isnull().sum().to_dict()
    total_nulls = df.isnull().sum().sum()
    return {"null_counts": null_counts, "total_nulls": int(total_nulls)}


def get_correlation(df: pd.DataFrame, params: dict) -> dict:
    """Calculate Pearson correlation between two numeric columns.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column1' (str) and 'column2' (str).

    Returns:
        dict: Contains 'column1', 'column2', and 'correlation' (float from -1 to 1).
              Returns {'error': str} if columns not found.

    Example:
        >>> df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [2, 4, 6, 8]})
        >>> get_correlation(df, {'column1': 'x', 'column2': 'y'})
        {'column1': 'x', 'column2': 'y', 'correlation': 1.0}
    """
    col1 = params.get("column1")
    col2 = params.get("column2")

    if col1 not in df.columns:
        return {"error": f"Column '{col1}' not found"}
    if col2 not in df.columns:
        return {"error": f"Column '{col2}' not found"}

    corr = df[col1].corr(df[col2])
    return {"column1": col1, "column2": col2, "correlation": corr}


def get_top_n_rows(df: pd.DataFrame, params: dict) -> dict:
    """Get the N rows with the highest values in a column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) and optional 'n' (int, default: 5).

    Returns:
        dict: Contains 'column', 'n', and 'top_rows' (list of record dicts).
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'name': ['A', 'B', 'C'], 'score': [85, 92, 78]})
        >>> get_top_n_rows(df, {'column': 'score', 'n': 2})
        {'column': 'score', 'n': 2, 'top_rows': [{'name': 'B', 'score': 92}, {'name': 'A', 'score': 85}]}
    """
    column = params.get("column")
    n = params.get("n", 5)

    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    top_rows = df.nlargest(n, column).to_dict(orient="records")
    return {"column": column, "n": n, "top_rows": top_rows}


def get_bottom_n_rows(df: pd.DataFrame, params: dict) -> dict:
    """Get the N rows with the lowest values in a column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) and optional 'n' (int, default: 5).

    Returns:
        dict: Contains 'column', 'n', and 'bottom_rows' (list of record dicts).
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'name': ['A', 'B', 'C'], 'score': [85, 92, 78]})
        >>> get_bottom_n_rows(df, {'column': 'score', 'n': 2})
        {'column': 'score', 'n': 2, 'bottom_rows': [{'name': 'C', 'score': 78}, {'name': 'A', 'score': 85}]}
    """
    column = params.get("column")
    n = params.get("n", 5)

    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    bottom_rows = df.nsmallest(n, column).to_dict(orient="records")
    return {"column": column, "n": n, "bottom_rows": bottom_rows}


def get_percentile(df: pd.DataFrame, params: dict) -> dict:
    """Calculate a specific percentile of a column.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with 'column' (str) and 'percentile' (number, 0-100).

    Returns:
        dict: Contains 'column', 'percentile', and 'value'.
              Returns {'error': str} if column not found.

    Example:
        >>> df = pd.DataFrame({'value': [10, 20, 30, 40, 50]})
        >>> get_percentile(df, {'column': 'value', 'percentile': 50})
        {'column': 'value', 'percentile': 50, 'value': 30.0}
    """
    column = params.get("column")
    percentile = params.get("percentile", 50)

    if column not in df.columns:
        return {"error": f"Column '{column}' not found"}

    value = df[column].quantile(percentile / 100)
    return {"column": column, "percentile": percentile, "value": value}


def search_value(df: pd.DataFrame, params: dict) -> dict:
    """Search for a value across all columns in the DataFrame.

    Args:
        df: The pandas DataFrame to search.
        params: Dict with 'value' (str or number) - the value to find.

    Returns:
        dict: Contains 'search_value' and 'matches' (list of {column, rows}).
              Each match shows up to 10 row indices.

    Example:
        >>> df = pd.DataFrame({'a': [1, 2, 1], 'b': ['x', 'y', 'z']})
        >>> search_value(df, {'value': 1})
        {'search_value': 1, 'matches': [{'column': 'a', 'rows': [0, 2]}]}
    """
    value = params.get("value")

    matches = []
    for col in df.columns:
        mask = df[col] == value
        if mask.any():
            matching_rows = df[mask].index.tolist()
            matches.append({"column": col, "rows": matching_rows[:10]})

    return {"search_value": value, "matches": matches}


def filter_and_aggregate(df: pd.DataFrame, params: dict) -> dict:
    """Filter rows by a condition and then calculate an aggregate statistic.

    This is useful for questions like "What is the average salary of engineers?"
    or "What is the total sales for product X?".

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with:
            - 'filter_column' (str): Column to filter by
            - 'filter_value' (any): Value to match in filter_column
            - 'agg_column' (str): Column to aggregate
            - 'agg_func' (str): One of 'mean', 'sum', 'min', 'max', 'count', 'median', 'std'

    Returns:
        dict: Contains filter info, aggregation info, matching row count, and result.
              Returns {'error': str} if columns not found.

    Example:
        >>> df = pd.DataFrame({
        ...     'dept': ['sales', 'sales', 'eng', 'eng'],
        ...     'salary': [50000, 60000, 80000, 90000]
        ... })
        >>> filter_and_aggregate(df, {
        ...     'filter_column': 'dept',
        ...     'filter_value': 'eng',
        ...     'agg_column': 'salary',
        ...     'agg_func': 'mean'
        ... })
        {'filter_column': 'dept', 'filter_value': 'eng', 'agg_column': 'salary',
         'agg_func': 'mean', 'matching_rows': 2, 'result': 85000.0}
    """
    filter_column = params.get("filter_column")
    filter_value = params.get("filter_value")
    agg_column = params.get("agg_column")
    agg_func = params.get("agg_func", "mean")

    if filter_column not in df.columns:
        return {"error": f"Column '{filter_column}' not found"}
    if agg_column not in df.columns:
        return {"error": f"Column '{agg_column}' not found"}

    # Filter the dataframe
    filtered_df = df[df[filter_column] == filter_value]

    if len(filtered_df) == 0:
        return {
            "filter_column": filter_column,
            "filter_value": filter_value,
            "agg_column": agg_column,
            "agg_func": agg_func,
            "matching_rows": 0,
            "result": None,
            "message": f"No rows found where {filter_column} = {filter_value}"
        }

    # Calculate the aggregate
    agg_functions = {
        "mean": filtered_df[agg_column].mean,
        "sum": filtered_df[agg_column].sum,
        "min": filtered_df[agg_column].min,
        "max": filtered_df[agg_column].max,
        "count": lambda: len(filtered_df),
        "median": filtered_df[agg_column].median,
        "std": filtered_df[agg_column].std,
    }

    if agg_func not in agg_functions:
        return {"error": f"Invalid agg_func '{agg_func}'. Use: mean, sum, min, max, count, median, std"}

    result = agg_functions[agg_func]()

    return {
        "filter_column": filter_column,
        "filter_value": filter_value,
        "agg_column": agg_column,
        "agg_func": agg_func,
        "matching_rows": len(filtered_df),
        "result": result
    }


def filter_multiple_conditions(df: pd.DataFrame, params: dict) -> dict:
    """Filter rows using multiple AND conditions.

    Supports both equality and comparison conditions on different columns.
    All conditions must be satisfied (AND logic).

    Args:
        df: The pandas DataFrame to filter.
        params: Dict with 'conditions' - a list of condition dicts, each with:
                - 'column' (str): column name
                - 'operator' (str): 'eq', 'gt', 'lt', 'gte', 'lte'
                - 'value': the value to compare against

    Returns:
        dict: Contains 'conditions', 'matching_rows' count, and 'sample' (first 5 rows).
              Returns {'error': str} if any column not found or invalid operator.

    Example:
        >>> df = pd.DataFrame({'hp': [100, 200, 150], 'wt': [2.5, 3.5, 2.8]})
        >>> filter_multiple_conditions(df, {
        ...     'conditions': [
        ...         {'column': 'hp', 'operator': 'gt', 'value': 100},
        ...         {'column': 'wt', 'operator': 'lt', 'value': 3.0}
        ...     ]
        ... })
        {'conditions': [...], 'matching_rows': 1, 'sample': [...]}
    """
    conditions = params.get("conditions", [])
    
    if not conditions:
        return {"error": "No conditions provided"}
    
    # Start with all rows
    mask = pd.Series([True] * len(df), index=df.index)
    
    operators = {
        "eq": lambda col, val: df[col] == val,
        "gt": lambda col, val: df[col] > val,
        "lt": lambda col, val: df[col] < val,
        "gte": lambda col, val: df[col] >= val,
        "lte": lambda col, val: df[col] <= val,
    }
    
    for cond in conditions:
        column = cond.get("column")
        operator = cond.get("operator", "eq")
        value = cond.get("value")
        
        if column not in df.columns:
            return {"error": f"Column '{column}' not found"}
        
        if operator not in operators:
            return {"error": f"Invalid operator '{operator}'. Use: eq, gt, lt, gte, lte"}
        
        mask = mask & operators[operator](column, value)
    
    filtered = df[mask]
    
    return {
        "conditions": conditions,
        "matching_rows": len(filtered),
        "sample": filtered.head(5).to_dict(orient="records")
    }


def compute_column_ratio(df: pd.DataFrame, params: dict) -> dict:
    """Compute the ratio between two columns and find max/min.

    Divides numerator_column by denominator_column and returns statistics
    about the computed ratio.

    Args:
        df: The pandas DataFrame to analyze.
        params: Dict with:
                - 'numerator_column' (str): column to use as numerator
                - 'denominator_column' (str): column to use as denominator
                - 'operation' (str, optional): 'max', 'min', 'mean' (default: 'max')

    Returns:
        dict: Contains 'numerator', 'denominator', 'operation', 'value' (the result),
              and 'row' (the row with that value, for max/min).
              Returns {'error': str} if columns not found.

    Example:
        >>> df = pd.DataFrame({'hp': [100, 200, 150], 'wt': [2.5, 4.0, 2.0]})
        >>> compute_column_ratio(df, {
        ...     'numerator_column': 'hp',
        ...     'denominator_column': 'wt',
        ...     'operation': 'max'
        ... })
        {'numerator': 'hp', 'denominator': 'wt', 'operation': 'max', 'value': 75.0, 'row': {...}}
    """
    numerator = params.get("numerator_column")
    denominator = params.get("denominator_column")
    operation = params.get("operation", "max")
    
    if numerator not in df.columns:
        return {"error": f"Column '{numerator}' not found"}
    if denominator not in df.columns:
        return {"error": f"Column '{denominator}' not found"}
    
    # Compute the ratio
    ratio = df[numerator] / df[denominator]
    
    if operation == "max":
        idx = ratio.idxmax()
        value = ratio[idx]
    elif operation == "min":
        idx = ratio.idxmin()
        value = ratio[idx]
    elif operation == "mean":
        value = ratio.mean()
        idx = None
    else:
        return {"error": f"Invalid operation '{operation}'. Use: max, min, mean"}
    
    result = {
        "numerator": numerator,
        "denominator": denominator,
        "operation": operation,
        "value": value,
    }
    
    if idx is not None:
        result["row"] = df.loc[idx].to_dict()
    
    return result


# Registro de todas las funciones disponibles
FUNCTIONS_REGISTRY = {
    "find_max_value_location": find_max_value_location,
    "find_min_value_location": find_min_value_location,
    "get_column_max": get_column_max,
    "get_column_min": get_column_min,
    "get_column_sum": get_column_sum,
    "get_column_mean": get_column_mean,
    "get_column_median": get_column_median,
    "get_column_std": get_column_std,
    "count_rows": count_rows,
    "count_columns": count_columns,
    "count_unique": count_unique,
    "get_value_counts": get_value_counts,
    "filter_by_value": filter_by_value,
    "filter_by_comparison": filter_by_comparison,
    "filter_multiple_conditions": filter_multiple_conditions,
    "filter_and_aggregate": filter_and_aggregate,
    "group_aggregate": group_aggregate,
    "compare_columns": compare_columns,
    "get_column_names": get_column_names,
    "describe_dataframe": describe_dataframe,
    "get_null_counts": get_null_counts,
    "get_correlation": get_correlation,
    "get_top_n_rows": get_top_n_rows,
    "get_bottom_n_rows": get_bottom_n_rows,
    "get_percentile": get_percentile,
    "search_value": search_value,
    "compute_column_ratio": compute_column_ratio,
}
