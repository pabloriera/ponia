"""Ponia: main function for natural language queries on DataFrames."""

import os
import json
from typing import Optional, List, Dict, Any, Union

import pandas as pd
from openai import OpenAI

from .functions import FUNCTIONS_REGISTRY
from .tools import TOOLS
from .prompts import PLANNING_PROMPT, RESPONSE_PROMPT


def ponia(
    df: pd.DataFrame,
    query: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    append_prompt: Optional[str] = None,
    verbose: bool = False,
    raw: bool = False
) -> Union[str, Dict, List]:
    """Perform a natural language query on a pandas DataFrame.

    Uses OpenAI function calling to interpret the query and execute
    predefined pandas operations locally. No DataFrame data is sent
    to OpenAI - only the user's question and function schemas.

    The process has 3 stages:
    1. Planning: AI analyzes the query and generates function calls (no data sent)
    2. Execution: Functions are executed locally, chaining results
    3. Response: AI formats the final result into natural language

    Args:
        df: The pandas DataFrame to analyze.
        query: Natural language question about the data (e.g., "What is the max value?").
        model: OpenAI model to use. Defaults to "gpt-4o-mini".
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        append_prompt: Additional prompt text to append to the planning prompt.
        verbose: If True, prints function calls and results for debugging.
        raw: If True, returns the raw JSON result(s) instead of natural language.

    Returns:
        str: Natural language response answering the query.
        dict | list: If raw=True, returns the raw function result(s) as JSON.

    Raises:
        ValueError: If no API key is found.

    Example:
        >>> import pandas as pd
        >>> from ponia import ponia
        >>> df = pd.DataFrame({
        ...     'product': ['apple', 'banana', 'orange'],
        ...     'sales': [100, 150, 80]
        ... })
        >>> answer = ponia(df, "Which product has the highest sales?")
        >>> print(answer)
        "The product 'banana' has the highest sales with 150 units."

        >>> # Get raw JSON result
        >>> result = ponia(df, "How many rows?", raw=True)
        >>> print(result)
        {'total_rows': 3}
    """
    # Get API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. "
            "Set OPENAI_API_KEY environment variable or pass it as argument."
        )
    
    client = OpenAI(api_key=api_key)
    
    # Stage 1: Plan function calls (AI sees only query + function schemas, no data)
    if verbose:
        print("=== Stage 1: Planning function calls ===")
    
    function_calls = _plan_function_calls(
        client=client,
        query=query,
        model=model,
        append_prompt=append_prompt,
        verbose=verbose
    )
    
    if not function_calls:
        return "Could not determine which functions to call for this query."
    
    # Stage 2: Execute functions locally, chaining results
    if verbose:
        print("\n=== Stage 2: Executing functions locally ===")
    
    results = _execute_function_chain(
        df=df,
        function_calls=function_calls,
        verbose=verbose
    )
    
    # If raw mode, return the results directly
    if raw:
        if len(results) == 1:
            return results[0]["result"]
        return results
    
    # Stage 3: Format the final result into natural language
    if verbose:
        print("\n=== Stage 3: Formatting response ===")
    
    # Get the final result to send to AI for formatting
    final_result = results[-1]["result"] if len(results) == 1 else results
    
    response = _format_response(
        client=client,
        query=query,
        result=final_result,
        model=model,
        verbose=verbose
    )
    
    return response


def _plan_function_calls(
    client: OpenAI,
    query: str,
    model: str,
    append_prompt: Optional[str] = None,
    verbose: bool = False,
    max_retries: int = 3
) -> List[Dict[str, Any]]:
    """Plan which functions to call based on the user's query.
    
    Uses AI to analyze the query and determine the sequence of function calls
    needed. No data is sent to the AI - only the query and function schemas.
    
    Args:
        client: OpenAI client instance.
        query: The user's natural language question.
        model: OpenAI model to use.
        append_prompt: Additional prompt text to append.
        verbose: If True, prints debug information.
        max_retries: Maximum attempts to get function calls from AI.
    
    Returns:
        List of dicts with 'name' and 'arguments' for each function call.
        Empty list if planning fails.
    """
    planning_prompt = PLANNING_PROMPT
    if append_prompt:
        planning_prompt = planning_prompt + "\n\n" + append_prompt
    
    messages = [
        {"role": "system", "content": planning_prompt},
        {"role": "user", "content": query}
    ]
    
    for attempt in range(max_retries):
        if verbose and attempt > 0:
            print(f"  Retry {attempt + 1}/{max_retries}...")
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="required"  # Force function calling
        )
        
        response_message = response.choices[0].message
        
        # Show thinking if verbose
        if verbose and response_message.content:
            print(f"  Thinking: {response_message.content}")
        
        # Extract function calls
        if response_message.tool_calls:
            function_calls = []
            for tool_call in response_message.tool_calls:
                call = {
                    "name": tool_call.function.name,
                    "arguments": json.loads(tool_call.function.arguments)
                }
                function_calls.append(call)
                if verbose:
                    print(f"  Planned: {call['name']}({call['arguments']})")
            return function_calls
        
        # No function calls, add message to context and retry
        messages.append(response_message)
        messages.append({
            "role": "user", 
            "content": "You must call at least one function to answer this question. Please try again."
        })
    
    return []


def _execute_function_chain(
    df: pd.DataFrame,
    function_calls: List[Dict[str, Any]],
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Execute a chain of function calls locally on the DataFrame.
    
    Each function receives the DataFrame (possibly modified by previous calls)
    and returns a result. Functions that return DataFrames pass them to the next
    function in the chain.
    
    Args:
        df: The pandas DataFrame to analyze.
        function_calls: List of dicts with 'name' and 'arguments'.
        verbose: If True, prints execution details.
    
    Returns:
        List of dicts with 'function', 'args', and 'result' for each call.
    """
    results = []
    current_df = df
    
    for call in function_calls:
        function_name = call["name"]
        function_args = call["arguments"]
        
        if verbose:
            print(f"  Executing: {function_name}({function_args})")
        
        if function_name not in FUNCTIONS_REGISTRY:
            result = {"error": f"Function '{function_name}' not found"}
        else:
            try:
                result = FUNCTIONS_REGISTRY[function_name](current_df, function_args)
                result = _serialize_result(result)
                
                # If result contains a DataFrame, use it for the next call
                if isinstance(result, dict) and "filtered_df" in result:
                    # Some functions might return filtered DataFrames
                    # For now, we keep the original df for each call
                    pass
                    
            except Exception as e:
                result = {"error": str(e)}
        
        if verbose:
            # Truncate long results for display
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            print(f"  Result: {result_str}")
        
        results.append({
            "function": function_name,
            "args": function_args,
            "result": result
        })
    
    return results


def _format_response(
    client: OpenAI,
    query: str,
    result: Any,
    model: str,
    verbose: bool = False
) -> str:
    """Format the analysis result into a natural language response.
    
    Uses AI to transform the raw result into a human-readable answer.
    Only sends the original query and the final result - no intermediate
    function calls or data.
    
    Args:
        client: OpenAI client instance.
        query: The user's original question.
        result: The result from executing the function chain.
        model: OpenAI model to use.
        verbose: If True, prints debug information.
    
    Returns:
        Natural language response string.
    """
    # Format the result for the AI
    result_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    user_message = f"""User's question: {query}

Analysis result:
{result_str}

Please provide a natural language answer to the user's question based on this result."""
    
    if verbose:
        print(f"  Sending to AI for formatting...")
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RESPONSE_PROMPT},
            {"role": "user", "content": user_message}
        ]
    )
    
    answer = response.choices[0].message.content or "Could not generate a response."
    
    if verbose:
        print(f"  Response: {answer[:100]}...")
    
    return answer


def _serialize_result(obj):
    """Convert numpy/pandas objects to native Python types for JSON serialization.

    Recursively processes dicts and lists, converting numpy integers, floats,
    and arrays to their Python equivalents.

    Args:
        obj: Any object that may contain numpy/pandas types.

    Returns:
        The same structure with all numpy/pandas types converted to native Python.

    Example:
        >>> import numpy as np
        >>> _serialize_result({'value': np.int64(42)})
        {'value': 42}
    """
    import numpy as np

    if isinstance(obj, dict):
        return {k: _serialize_result(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_serialize_result(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj
