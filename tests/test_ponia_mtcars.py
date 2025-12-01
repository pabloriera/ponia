"""Integration tests for ponia using the mtcars dataset.

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


def extract_result(result, target_key=None, target_function=None):
    """Extract the actual result from ponia's raw output.
    
    Args:
        result: The raw result from ponia (dict or list)
        target_key: Key to look for in the result (e.g., 'max_value', 'matching_rows')
        target_function: Function name to find in list results
    
    Returns:
        The extracted result dict
    """
    if isinstance(result, list):
        # If looking for a specific function, find it
        if target_function:
            for r in result:
                if r.get('function') == target_function:
                    return r.get('result', r)
        # If looking for a specific key, find first result with that key
        if target_key:
            for r in result:
                res = r.get('result', r)
                if target_key in res:
                    return res
        # Default: return last result
        last = result[-1]
        return last.get('result', last)
    return result


@pytest.fixture
def mtcars_df():
    """Load the mtcars dataset."""
    url = "https://raw.githubusercontent.com/selva86/datasets/master/mtcars.csv"
    return pd.read_csv(url)


class TestBasicCounting:
    """Tests for basic counting operations."""

    def test_count_rows(self, mtcars_df):
        """1. How many rows does the dataset have (i.e., how many cars)?"""
        # Explicit pandas
        expected = len(mtcars_df)
        
        # Ponia
        result = ponia(
            mtcars_df,
            "How many rows does the dataset have?",
            raw=True
        )
        
        result = extract_result(result, target_key='total_rows')
        assert result['total_rows'] == expected

    def test_count_cyl_4(self, mtcars_df):
        """2. How many cars have cyl == 4?"""
        # Explicit pandas
        expected = len(mtcars_df[mtcars_df['cyl'] == 4])
        
        # Ponia
        result = ponia(
            mtcars_df,
            "How many cars have cyl equal to 4?",
            raw=True
        )
        
        result = extract_result(result, target_key='matching_rows', target_function='filter_by_value')
        assert result['matching_rows'] == expected


class TestColumnStatistics:
    """Tests for column statistics."""

    def test_max_mpg(self, mtcars_df):
        """3. What is the maximum value of mpg?"""
        # Explicit pandas
        expected = mtcars_df['mpg'].max()
        
        # Ponia
        result = ponia(
            mtcars_df,
            "What is the maximum value of mpg?",
            raw=True
        )
        
        result = extract_result(result, target_key='max_value')
        assert result['max_value'] == expected

    def test_mean_hp(self, mtcars_df):
        """4. What is the mean value of hp?"""
        # Explicit pandas
        expected = mtcars_df['hp'].mean()
        
        # Ponia
        result = ponia(
            mtcars_df,
            "What is the mean value of hp?",
            raw=True
        )
        
        result = extract_result(result, target_key='mean')
        assert abs(result['mean'] - expected) < 0.01

    def test_min_carb(self, mtcars_df):
        """8. What is the minimum value of carb?"""
        # Explicit pandas
        expected = mtcars_df['carb'].min()
        
        # Ponia
        result = ponia(
            mtcars_df,
            "What is the minimum value of carb?",
            raw=True
        )
        
        result = extract_result(result, target_key='min_value')
        assert result['min_value'] == expected


class TestTopBottom:
    """Tests for top/bottom N operations."""

    def test_bottom_5_by_wt(self, mtcars_df):
        """5. Which 5 cars have the smallest values of wt?"""
        # Explicit pandas
        expected = mtcars_df.nsmallest(5, 'wt')['wt'].tolist()
        
        # Ponia
        result = ponia(
            mtcars_df,
            "What are the bottom 5 rows sorted by wt?",
            raw=True
        )
        
        result = extract_result(result, target_key='bottom_rows')
        result_wt = [r['wt'] for r in result['bottom_rows']]
        for i, val in enumerate(expected):
            assert abs(result_wt[i] - val) < 0.01


class TestFiltering:
    """Tests for filtering operations."""

    def test_count_am_0(self, mtcars_df):
        """9. How many cars have am == 0 (automatic transmission)?"""
        # Explicit pandas
        expected = len(mtcars_df[mtcars_df['am'] == 0])
        
        # Ponia
        result = ponia(
            mtcars_df,
            "How many cars have am equal to 0?",
            raw=True
        )
        
        result = extract_result(result, target_key='matching_rows', target_function='filter_by_value')
        assert result['matching_rows'] == expected


class TestCorrelation:
    """Tests for correlation operations."""

    def test_correlation_mpg_wt(self, mtcars_df):
        """10. What is the Pearson correlation between mpg and wt?"""
        # Explicit pandas
        expected = mtcars_df['mpg'].corr(mtcars_df['wt'])
        
        # Ponia
        result = ponia(
            mtcars_df,
            "What is the correlation between mpg and wt?",
            raw=True
        )
        
        result = extract_result(result, target_key='correlation')
        assert abs(result['correlation'] - expected) < 0.01


class TestFilterAndAggregate:
    """Tests for filter + aggregate operations."""

    def test_mean_mpg_for_cyl_6(self, mtcars_df):
        """13. What is the mean value of mpg among cars with cyl == 6?"""
        # Explicit pandas
        expected = mtcars_df[mtcars_df['cyl'] == 6]['mpg'].mean()
        
        # Ponia
        result = ponia(
            mtcars_df,
            "What is the mean mpg for cars with cyl equal to 6? Use filter_and_aggregate.",
            raw=True
        )
        
        result = extract_result(result, target_function='filter_and_aggregate')
        assert abs(result['result'] - expected) < 0.01


class TestComplexQueries:
    """Tests for more complex queries that may require multiple operations."""

    def test_hp_gt_150_and_mpg_gt_20(self, mtcars_df):
        """6. How many cars satisfy hp > 150 and mpg > 20?"""
        # Explicit pandas
        expected = len(mtcars_df[(mtcars_df['hp'] > 150) & (mtcars_df['mpg'] > 20)])
        
        # Ponia
        result = ponia(
            mtcars_df,
            "How many cars have hp greater than 150 and mpg greater than 20? Use filter_multiple_conditions.",
            raw=True
        )
        
        result = extract_result(result, target_key='matching_rows', target_function='filter_multiple_conditions')
        assert result['matching_rows'] == expected

    def test_hp_gt_100_and_wt_lt_3(self, mtcars_df):
        """12. How many cars satisfy hp > 100 and wt < 3.0?"""
        # Explicit pandas
        expected = len(mtcars_df[(mtcars_df['hp'] > 100) & (mtcars_df['wt'] < 3.0)])
        
        # Ponia
        result = ponia(
            mtcars_df,
            "How many cars have hp greater than 100 and wt less than 3.0? Use filter_multiple_conditions.",
            raw=True
        )
        
        result = extract_result(result, target_key='matching_rows', target_function='filter_multiple_conditions')
        assert result['matching_rows'] == expected


class TestIdxMaxQueries:
    """Tests for finding rows with max/min values."""

    def test_car_with_max_wt(self, mtcars_df):
        """7. Which car has the maximum value of wt?"""
        # Explicit pandas
        expected_wt = mtcars_df['wt'].max()
        
        # Ponia
        result = ponia(
            mtcars_df,
            "What is the maximum value of wt?",
            raw=True
        )
        
        result = extract_result(result, target_key='max_value')
        assert abs(result['max_value'] - expected_wt) < 0.01

    def test_car_with_max_hp_wt_ratio(self, mtcars_df):
        """11. Which car has the largest value of the ratio hp / wt?"""
        # Explicit pandas
        ratio = mtcars_df['hp'] / mtcars_df['wt']
        expected_ratio = ratio.max()
        expected_idx = ratio.idxmax()
        expected_row = mtcars_df.loc[expected_idx]
        
        # Ponia
        result = ponia(
            mtcars_df,
            "Which row has the maximum value of hp divided by wt? Use compute_column_ratio.",
            raw=True
        )
        
        result = extract_result(result, target_function='compute_column_ratio')
        assert abs(result['value'] - expected_ratio) < 0.01
        assert abs(result['row']['hp'] - expected_row['hp']) < 0.01
        assert abs(result['row']['wt'] - expected_row['wt']) < 0.01
