"""Tests para las funciones de ponia usando datasets clásicos."""

import pytest
import pandas as pd
import seaborn as sns

from ponia.functions import (
    find_max_value_location,
    find_min_value_location,
    get_column_max,
    get_column_min,
    get_column_sum,
    get_column_mean,
    get_column_median,
    get_column_std,
    count_rows,
    count_columns,
    count_unique,
    get_value_counts,
    filter_by_value,
    filter_by_comparison,
    group_aggregate,
    compare_columns,
    get_column_names,
    describe_dataframe,
    get_null_counts,
    get_correlation,
    get_top_n_rows,
    get_bottom_n_rows,
    get_percentile,
    search_value,
)


# Fixtures para datasets clásicos
@pytest.fixture
def iris():
    """Dataset iris de seaborn."""
    return sns.load_dataset("iris")


@pytest.fixture
def tips():
    """Dataset tips de seaborn."""
    return sns.load_dataset("tips")


@pytest.fixture
def titanic():
    """Dataset titanic de seaborn."""
    return sns.load_dataset("titanic")


@pytest.fixture
def simple_df():
    """DataFrame simple para tests básicos."""
    return pd.DataFrame({
        "nombre": ["Ana", "Bob", "Carlos", "Diana"],
        "edad": [25, 30, 35, 28],
        "salario": [50000, 60000, 75000, 55000],
        "ciudad": ["Madrid", "Barcelona", "Madrid", "Valencia"]
    })


# Tests de funciones de máximo/mínimo
class TestMaxMinFunctions:
    
    def test_find_max_value_location_iris(self, iris):
        result = find_max_value_location(iris, {})
        assert "value" in result
        assert "column" in result
        assert "row" in result
        assert result["value"] == iris.select_dtypes(include='number').max().max()
    
    def test_find_min_value_location_iris(self, iris):
        result = find_min_value_location(iris, {})
        assert "value" in result
        assert result["value"] == iris.select_dtypes(include='number').min().min()
    
    def test_get_column_max(self, tips):
        result = get_column_max(tips, {"column": "total_bill"})
        assert result["max_value"] == tips["total_bill"].max()
        assert result["column"] == "total_bill"
    
    def test_get_column_min(self, tips):
        result = get_column_min(tips, {"column": "tip"})
        assert result["min_value"] == tips["tip"].min()
    
    def test_get_column_max_not_found(self, tips):
        result = get_column_max(tips, {"column": "inexistente"})
        assert "error" in result


# Tests de funciones estadísticas
class TestStatsFunctions:
    
    def test_get_column_sum(self, simple_df):
        result = get_column_sum(simple_df, {"column": "salario"})
        assert result["sum"] == 240000
    
    def test_get_column_mean(self, iris):
        result = get_column_mean(iris, {"column": "sepal_length"})
        assert abs(result["mean"] - iris["sepal_length"].mean()) < 0.001
    
    def test_get_column_median(self, tips):
        result = get_column_median(tips, {"column": "total_bill"})
        assert result["median"] == tips["total_bill"].median()
    
    def test_get_column_std(self, iris):
        result = get_column_std(iris, {"column": "petal_width"})
        assert abs(result["std"] - iris["petal_width"].std()) < 0.001


# Tests de funciones de conteo
class TestCountFunctions:
    
    def test_count_rows(self, iris):
        result = count_rows(iris, {})
        assert result["total_rows"] == 150
    
    def test_count_columns(self, tips):
        result = count_columns(tips, {})
        assert result["total_columns"] == len(tips.columns)
    
    def test_count_unique(self, iris):
        result = count_unique(iris, {"column": "species"})
        assert result["unique_count"] == 3
        assert "setosa" in result["sample_values"]
    
    def test_get_value_counts(self, titanic):
        result = get_value_counts(titanic, {"column": "sex"})
        assert "value_counts" in result
        assert "male" in result["value_counts"]
        assert "female" in result["value_counts"]


# Tests de funciones de filtrado
class TestFilterFunctions:
    
    def test_filter_by_value(self, iris):
        result = filter_by_value(iris, {"column": "species", "value": "setosa"})
        assert result["matching_rows"] == 50
    
    def test_filter_by_comparison_gt(self, tips):
        result = filter_by_comparison(tips, {
            "column": "total_bill",
            "operator": "gt",
            "value": 40
        })
        expected = len(tips[tips["total_bill"] > 40])
        assert result["matching_rows"] == expected
    
    def test_filter_by_comparison_lte(self, simple_df):
        result = filter_by_comparison(simple_df, {
            "column": "edad",
            "operator": "lte",
            "value": 28
        })
        assert result["matching_rows"] == 2  # Ana (25) y Diana (28)


# Tests de funciones de agrupación
class TestGroupFunctions:
    
    def test_group_aggregate_mean(self, iris):
        result = group_aggregate(iris, {
            "group_by": "species",
            "agg_column": "sepal_length",
            "agg_func": "mean"
        })
        assert len(result["result"]) == 3
        assert "setosa" in result["result"]
    
    def test_group_aggregate_count(self, tips):
        result = group_aggregate(tips, {
            "group_by": "day",
            "agg_func": "count"
        })
        assert sum(result["result"].values()) == len(tips)
    
    def test_group_aggregate_sum(self, tips):
        result = group_aggregate(tips, {
            "group_by": "sex",
            "agg_column": "total_bill",
            "agg_func": "sum"
        })
        expected = tips.groupby("sex")["total_bill"].sum().to_dict()
        for key in expected:
            assert abs(result["result"][key] - expected[key]) < 0.01


# Tests de funciones de comparación y correlación
class TestComparisonFunctions:
    
    def test_compare_columns(self, iris):
        result = compare_columns(iris, {
            "column1": "sepal_length",
            "column2": "petal_length"
        })
        assert result["column1"]["name"] == "sepal_length"
        assert result["column2"]["name"] == "petal_length"
        assert result["column1"]["mean"] is not None
    
    def test_get_correlation(self, iris):
        result = get_correlation(iris, {
            "column1": "sepal_length",
            "column2": "petal_length"
        })
        expected = iris["sepal_length"].corr(iris["petal_length"])
        assert abs(result["correlation"] - expected) < 0.001


# Tests de funciones de información del DataFrame
class TestInfoFunctions:
    
    def test_get_column_names(self, tips):
        result = get_column_names(tips, {})
        assert "total_bill" in result["columns"]
        assert "tip" in result["columns"]
        assert len(result["dtypes"]) == len(tips.columns)
    
    def test_describe_dataframe(self, iris):
        result = describe_dataframe(iris, {})
        assert result["shape"]["rows"] == 150
        assert result["shape"]["columns"] == 5
        assert "statistics" in result
    
    def test_get_null_counts(self, titanic):
        result = get_null_counts(titanic, {})
        assert "null_counts" in result
        assert "total_nulls" in result
        # Titanic tiene valores nulos en varias columnas
        assert result["total_nulls"] > 0


# Tests de funciones top/bottom
class TestTopBottomFunctions:
    
    def test_get_top_n_rows(self, simple_df):
        result = get_top_n_rows(simple_df, {"column": "salario", "n": 2})
        assert len(result["top_rows"]) == 2
        assert result["top_rows"][0]["salario"] == 75000  # Carlos
    
    def test_get_bottom_n_rows(self, simple_df):
        result = get_bottom_n_rows(simple_df, {"column": "edad", "n": 2})
        assert len(result["bottom_rows"]) == 2
        assert result["bottom_rows"][0]["edad"] == 25  # Ana


# Tests de funciones de percentil y búsqueda
class TestPercentileSearchFunctions:
    
    def test_get_percentile(self, iris):
        result = get_percentile(iris, {"column": "sepal_length", "percentile": 50})
        expected = iris["sepal_length"].median()
        assert result["value"] == expected
    
    def test_get_percentile_90(self, tips):
        result = get_percentile(tips, {"column": "total_bill", "percentile": 90})
        expected = tips["total_bill"].quantile(0.90)
        assert abs(result["value"] - expected) < 0.01
    
    def test_search_value(self, simple_df):
        result = search_value(simple_df, {"value": "Madrid"})
        assert len(result["matches"]) > 0
        assert result["matches"][0]["column"] == "ciudad"
    
    def test_search_value_numeric(self, simple_df):
        result = search_value(simple_df, {"value": 30})
        assert any(m["column"] == "edad" for m in result["matches"])
