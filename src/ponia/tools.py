"""Tool definitions for OpenAI function calling.

Each tool defines a JSON Schema that OpenAI uses to decide which function to call.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "find_max_value_location",
            "description": "Find the global maximum value in the entire DataFrame and return its location (column and row). Useful for questions like 'what is the largest value?' or 'where is the maximum?'",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_min_value_location",
            "description": "Find the global minimum value in the entire DataFrame and return its location (column and row). Useful for questions like 'what is the smallest value?' or 'where is the minimum?'",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_max",
            "description": "Get the maximum value of a specific column and the row where it is located.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column to analyze"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_min",
            "description": "Get the minimum value of a specific column and the row where it is located.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column to analyze"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_sum",
            "description": "Calculate the total sum of all values in a numeric column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column to sum"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_mean",
            "description": "Calculate the average (arithmetic mean) of a numeric column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_median",
            "description": "Calculate the median of a numeric column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_std",
            "description": "Calculate the standard deviation of a numeric column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_rows",
            "description": "Count the total number of rows in the DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_columns",
            "description": "Count the total number of columns in the DataFrame.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_unique",
            "description": "Count how many unique values are in a column and show some examples.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_value_counts",
            "description": "Get the frequency count of each value in a column. Useful for seeing category distribution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column"
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Number of most frequent values to show (default: 10)"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_value",
            "description": "Filter rows where a column has a specific exact value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column to filter"
                    },
                    "value": {
                        "type": ["string", "number", "boolean"],
                        "description": "Exact value to search for"
                    }
                },
                "required": ["column", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_by_comparison",
            "description": "Filter rows by numeric comparison (greater than, less than, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column"
                    },
                    "operator": {
                        "type": "string",
                        "enum": ["gt", "lt", "gte", "lte"],
                        "description": "Operator: gt (>), lt (<), gte (>=), lte (<=)"
                    },
                    "value": {
                        "type": "number",
                        "description": "Value to compare against"
                    }
                },
                "required": ["column", "operator", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_and_aggregate",
            "description": "Filter rows where a column equals a value, then calculate a statistic on another column. Use this for questions like 'What is the average X for category Y?' or 'What is the total X where Y equals Z?'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filter_column": {
                        "type": "string",
                        "description": "Column to filter by (e.g., 'department', 'category', 'occupation')"
                    },
                    "filter_value": {
                        "type": ["string", "number", "boolean"],
                        "description": "Value to match in the filter column"
                    },
                    "agg_column": {
                        "type": "string",
                        "description": "Column to calculate the statistic on (e.g., 'salary', 'price', 'wage')"
                    },
                    "agg_func": {
                        "type": "string",
                        "enum": ["mean", "sum", "min", "max", "count", "median", "std"],
                        "description": "Aggregation function to apply"
                    }
                },
                "required": ["filter_column", "filter_value", "agg_column", "agg_func"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "group_aggregate",
            "description": "Group data by a column and apply an aggregation function (sum, average, count, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "description": "Column to group by"
                    },
                    "agg_column": {
                        "type": "string",
                        "description": "Column to aggregate (not required for 'count')"
                    },
                    "agg_func": {
                        "type": "string",
                        "enum": ["sum", "mean", "count", "min", "max"],
                        "description": "Aggregation function"
                    }
                },
                "required": ["group_by", "agg_func"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compare_columns",
            "description": "Compare basic statistics between two columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column1": {
                        "type": "string",
                        "description": "First column"
                    },
                    "column2": {
                        "type": "string",
                        "description": "Second column"
                    }
                },
                "required": ["column1", "column2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_names",
            "description": "List all columns in the DataFrame with their data types.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "describe_dataframe",
            "description": "Get a complete statistical summary of the DataFrame (similar to df.describe()).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_null_counts",
            "description": "Count null/missing values in each column.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_correlation",
            "description": "Calculate the Pearson correlation between two numeric columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column1": {
                        "type": "string",
                        "description": "First column"
                    },
                    "column2": {
                        "type": "string",
                        "description": "Second column"
                    }
                },
                "required": ["column1", "column2"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_top_n_rows",
            "description": "Get the N rows with the highest values in a specific column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column to sort by"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to return (default: 5)"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_bottom_n_rows",
            "description": "Get the N rows with the lowest values in a specific column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Column to sort by"
                    },
                    "n": {
                        "type": "integer",
                        "description": "Number of rows to return (default: 5)"
                    }
                },
                "required": ["column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_percentile",
            "description": "Calculate a specific percentile of a column (e.g., 90th percentile, median is 50th percentile).",
            "parameters": {
                "type": "object",
                "properties": {
                    "column": {
                        "type": "string",
                        "description": "Name of the column"
                    },
                    "percentile": {
                        "type": "number",
                        "description": "Percentile to calculate (0-100)"
                    }
                },
                "required": ["column", "percentile"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_value",
            "description": "Search for a specific value across the entire DataFrame and return where it is found.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": ["string", "number"],
                        "description": "Value to search for"
                    }
                },
                "required": ["value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "filter_multiple_conditions",
            "description": "Filter rows using multiple AND conditions. Use this when you need to filter by more than one condition simultaneously, e.g., 'hp > 100 AND wt < 3.0'. All conditions must be satisfied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "conditions": {
                        "type": "array",
                        "description": "List of conditions to apply (all must be true)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column": {
                                    "type": "string",
                                    "description": "Column name"
                                },
                                "operator": {
                                    "type": "string",
                                    "enum": ["eq", "gt", "lt", "gte", "lte"],
                                    "description": "Comparison operator: eq (equal), gt (>), lt (<), gte (>=), lte (<=)"
                                },
                                "value": {
                                    "type": ["string", "number"],
                                    "description": "Value to compare against"
                                }
                            },
                            "required": ["column", "operator", "value"]
                        }
                    }
                },
                "required": ["conditions"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "compute_column_ratio",
            "description": "Compute the ratio between two columns (numerator/denominator) and find the max, min, or mean of that ratio. Use for questions like 'which row has the maximum hp/wt ratio?' or 'what is the average of column A divided by column B?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "numerator_column": {
                        "type": "string",
                        "description": "Column to use as numerator"
                    },
                    "denominator_column": {
                        "type": "string",
                        "description": "Column to use as denominator"
                    },
                    "operation": {
                        "type": "string",
                        "enum": ["max", "min", "mean"],
                        "description": "Operation to perform on the computed ratio (default: max)"
                    }
                },
                "required": ["numerator_column", "denominator_column"]
            }
        }
    }
]
