"""Ponia: main function for natural language queries on DataFrames."""

import os
import json
from typing import Optional

import pandas as pd
from openai import OpenAI

from .functions import FUNCTIONS_REGISTRY
from .tools import TOOLS
from .prompts import SYSTEM_PROMPT


def ponia(
    df: pd.DataFrame,
    query: str,
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    append_prompt: Optional[str] = None,
    verbose: bool = False,
    raw: bool = False
) -> str:
    """Perform a natural language query on a pandas DataFrame.

    Uses OpenAI function calling to interpret the query and execute
    predefined pandas operations locally. No DataFrame data is sent
    to OpenAI - only the user's question and function schemas.

    Args:
        df: The pandas DataFrame to analyze.
        query: Natural language question about the data (e.g., "What is the max value?").
        model: OpenAI model to use. Defaults to "gpt-4o-mini".
        api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
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

        >>> # With verbose mode to see function calls
        >>> answer = ponia(df, "How many rows?", verbose=True)
        Thinking: I need to count the rows...
        Calling: count_rows({})
        Result: {'total_rows': 3}
        "The DataFrame has 3 rows."

        >>> # Get raw JSON result
        >>> result = ponia(df, "How many rows?", raw=True)
        >>> print(result)
        {'total_rows': 3}
    """
    # Get API key
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "No se encontró OPENAI_API_KEY. "
            "Configúrala como variable de entorno o pásala como argumento."
        )
    
    client = OpenAI(api_key=api_key)
    
    # If append_prompt is provided, add it to the System prompt
    if append_prompt:
        system_prompt = SYSTEM_PROMPT + "\n\n" + append_prompt
    else:
        system_prompt = SYSTEM_PROMPT
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    # First call: get function calls
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message

    # If there is text content (thinking), show it in verbose
    if verbose and response_message.content:
        print(f"\n{response_message.content}\n")

    # If there are no tool calls, return direct response
    if not response_message.tool_calls:
        return response_message.content or "Could not process the query."

    # Process tool calls
    messages.append(response_message)

    raw_results = []

    for tool_call in response_message.tool_calls:
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)

        if verbose:
            print(f"Calling: {function_name}({function_args})")

        # Execute function locally
        if function_name in FUNCTIONS_REGISTRY:
            try:
                result = FUNCTIONS_REGISTRY[function_name](df, function_args)
                # Convert numpy/pandas values to native Python types
                result = _serialize_result(result)
            except Exception as e:
                result = {"error": str(e)}
        else:
            result = {"error": f"Function '{function_name}' not found"}

        if verbose:
            print(f"Result: {result}\n")

        raw_results.append({"function": function_name, "args": function_args, "result": result})

        # Add result to context
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result, ensure_ascii=False)
        })

    # Si raw=True, retornar resultados JSON sin procesar
    if raw:
        if len(raw_results) == 1:
            return raw_results[0]["result"]
        return raw_results
    
    # Second call: get final response
    final_response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    return final_response.choices[0].message.content or "Could not generate a response."


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
