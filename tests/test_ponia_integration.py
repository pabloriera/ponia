"""Integration tests for ponia using the wages dataset.

These tests compare ponia's raw results against explicit pandas operations
to verify that the AI is calling the correct functions with correct parameters.
"""

import pytest
import pandas as pd
import os

# Skip all tests if no API key is available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set"
)

from ponia import ponia


@pytest.fixture
def wages_df():
    """Load the wages 1985 dataset."""
    url = "https://raw.githubusercontent.com/stdlib-js/datasets-berndt-cps-wages-1985/refs/heads/main/data/data.csv"
    return pd.read_csv(url)


class TestFilterAndAggregate:
    """Tests for filter + aggregate operations."""

    def test_average_wage_for_professional(self, wages_df):
        """Average wage for workers with occupation professional."""
        # Explicit pandas
        expected = wages_df[wages_df['occupation'] == 'professional']['wage'].mean()
        
        # Ponia
        result = ponia(
            wages_df,
            "¿Cuál es el wage promedio de los trabajadores con occupation professional?",
            raw=True
        )
        
        assert abs(result['result'] - expected) < 0.01

    def test_max_wage_for_union_members(self, wages_df):
        """Maximum wage for union members (union=1)."""
        # Explicit pandas
        expected = wages_df[wages_df['union'] == 1]['wage'].max()
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the maximum wage for workers where union is 1?",
            raw=True
        )
        
        # Handle case where result might be a list (multiple function calls)
        if isinstance(result, list):
            result = result[-1]['result']
        
        assert abs(result['result'] - expected) < 0.01

    def test_total_education_years_for_south(self, wages_df):
        """Sum of education years for workers in south region (south=1)."""
        # Explicit pandas
        expected = wages_df[wages_df['south'] == 1]['education'].sum()
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the sum of education for workers where south is 1?",
            raw=True
        )
        
        # Handle case where result might be a list
        if isinstance(result, list):
            result = result[-1]['result']
        
        assert abs(result['result'] - expected) < 0.01


class TestGroupAggregate:
    """Tests for group by + aggregate operations."""

    def test_average_wage_by_gender(self, wages_df):
        """Average wage grouped by gender."""
        # Explicit pandas
        expected = wages_df.groupby('gender')['wage'].mean().to_dict()
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the average wage grouped by gender?",
            raw=True
        )
        
        for key in expected:
            assert abs(result['result'][key] - expected[key]) < 0.01

    def test_count_by_occupation(self, wages_df):
        """Count of workers by occupation."""
        # Explicit pandas
        expected = wages_df.groupby('occupation').size().to_dict()
        
        # Ponia
        result = ponia(
            wages_df,
            "How many workers are there in each occupation? Group by occupation and count.",
            raw=True
        )
        
        for key in expected:
            assert result['result'][key] == expected[key]


class TestColumnStatistics:
    """Tests for single column statistics."""

    def test_mean_age(self, wages_df):
        """Average age of all workers."""
        # Explicit pandas
        expected = wages_df['age'].mean()
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the mean age?",
            raw=True
        )
        
        assert abs(result['mean'] - expected) < 0.01

    def test_median_wage(self, wages_df):
        """Median wage of all workers."""
        # Explicit pandas
        expected = wages_df['wage'].median()
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the median wage?",
            raw=True
        )
        
        assert abs(result['median'] - expected) < 0.01

    def test_max_experience(self, wages_df):
        """Maximum experience."""
        # Explicit pandas
        expected = wages_df['experience'].max()
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the maximum experience?",
            raw=True
        )
        
        assert result['max_value'] == expected

    def test_sum_education(self, wages_df):
        """Total years of education."""
        # Explicit pandas
        expected = wages_df['education'].sum()
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the sum of education?",
            raw=True
        )
        
        assert result['sum'] == expected


class TestCounting:
    """Tests for counting operations."""

    def test_count_rows(self, wages_df):
        """Total number of workers."""
        # Explicit pandas
        expected = len(wages_df)
        
        # Ponia
        result = ponia(
            wages_df,
            "How many rows are there?",
            raw=True
        )
        
        assert result['total_rows'] == expected

    def test_unique_occupations(self, wages_df):
        """Number of unique occupations."""
        # Explicit pandas
        expected = wages_df['occupation'].nunique()
        
        # Ponia
        result = ponia(
            wages_df,
            "How many unique values are in occupation?",
            raw=True
        )
        
        assert result['unique_count'] == expected


class TestFiltering:
    """Tests for filtering operations."""

    def test_filter_high_wage(self, wages_df):
        """Count workers with wage greater than 10."""
        # Explicit pandas
        expected = len(wages_df[wages_df['wage'] > 10])
        
        # Ponia
        result = ponia(
            wages_df,
            "How many workers have wage greater than 10?",
            raw=True
        )
        
        assert result['matching_rows'] == expected

    def test_filter_by_married(self, wages_df):
        """Count married workers (married=1)."""
        # Explicit pandas
        expected = len(wages_df[wages_df['married'] == 1])
        
        # Ponia
        result = ponia(
            wages_df,
            "How many workers have married equal to 1? Use filter_by_value.",
            raw=True
        )
        
        # Handle different response formats
        if isinstance(result, list):
            result = result[-1]['result']
        
        assert result['matching_rows'] == expected


class TestTopBottom:
    """Tests for top/bottom N operations."""

    def test_top_3_highest_wages(self, wages_df):
        """Top 3 workers by wage."""
        # Explicit pandas
        expected = wages_df.nlargest(3, 'wage')['wage'].tolist()
        
        # Ponia
        result = ponia(
            wages_df,
            "What are the top 3 rows by wage?",
            raw=True
        )
        
        result_wages = [r['wage'] for r in result['top_rows']]
        for i, val in enumerate(expected):
            assert abs(result_wages[i] - val) < 0.01

    def test_bottom_5_by_experience(self, wages_df):
        """Bottom 5 workers by experience."""
        # Explicit pandas
        expected = wages_df.nsmallest(5, 'experience')['experience'].tolist()
        
        # Ponia
        result = ponia(
            wages_df,
            "What are the bottom 5 rows by experience?",
            raw=True
        )
        
        result_exp = [r['experience'] for r in result['bottom_rows']]
        assert result_exp == expected


class TestPercentile:
    """Tests for percentile operations."""

    def test_90th_percentile_wage(self, wages_df):
        """90th percentile of wage."""
        # Explicit pandas
        expected = wages_df['wage'].quantile(0.90)
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the 90th percentile of wage?",
            raw=True
        )
        
        assert abs(result['value'] - expected) < 0.01

    def test_25th_percentile_age(self, wages_df):
        """25th percentile of age."""
        # Explicit pandas
        expected = wages_df['age'].quantile(0.25)
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the 25th percentile of age?",
            raw=True
        )
        
        assert abs(result['value'] - expected) < 0.01


class TestCorrelation:
    """Tests for correlation operations."""

    def test_correlation_education_wage(self, wages_df):
        """Correlation between education and wage."""
        # Explicit pandas
        expected = wages_df['education'].corr(wages_df['wage'])
        
        # Ponia
        result = ponia(
            wages_df,
            "What is the correlation between education and wage?",
            raw=True
        )
        
        assert abs(result['correlation'] - expected) < 0.01
