"""
Utility functions for parsing and cleaning text data.
"""

import json
import ast
from typing import Union, Dict, List

def parse_dict_strings(text: str) -> str:
    """
    Parse and clean dictionary strings from various formats.
    
    Args:
        text (str): Input text containing dictionary-like structure
        
    Returns:
        str: Cleaned and properly formatted dictionary string
    """
    # Replace smart quotes with regular quotes
    replacements = {
        ''': "'",  # Left single quote
        ''': "'",  # Right single quote
        '"': '"',  # Left double quote
        '"': '"',  # Right double quote
    }
    
    # Replace all smart quotes
    for smart_quote, regular_quote in replacements.items():
        text = text.replace(smart_quote, regular_quote)
    
    try:
        # Try parsing as JSON first
        data = json.loads(text.replace("'", '"'))
        return _clean_parsed_data(data)
            
    except json.JSONDecodeError:
        return _manual_dict_parse(text)

def _clean_parsed_data(data: Union[List, Dict]) -> str:
    """Clean parsed JSON data."""
    if isinstance(data, list):
        return str([{k: str(v).replace("'", "\\'") for k, v in item.items()} 
                   for item in data])
    elif isinstance(data, dict):
        return str({k: str(v).replace("'", "\\'") for k, v in data.items()})
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")

def _manual_dict_parse(text: str) -> str:
    """Manually parse dictionary-like strings."""
    try:
        if not (text.startswith("{") and text.endswith("}")):
            return text
            
        content = text[1:-1].strip()
        pairs = _split_dict_pairs(content)
        result_dict = _process_dict_pairs(pairs)
        
        return "{" + ", ".join(f"'{k}': '{v}'" for k, v in result_dict.items()) + "}"
        
    except Exception as e:
        print(f"Error in manual parsing: {str(e)}")
        return text

def _split_dict_pairs(content: str) -> List[str]:
    """Split dictionary content into key-value pairs."""
    pairs = []
    current_pair = []
    in_quotes = False
    quote_char = None
    
    for char in content:
        if char in ['"', "'"]:
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
        elif char == ',' and not in_quotes:
            pairs.append(''.join(current_pair).strip())
            current_pair = []
            continue
        current_pair.append(char)
    
    if current_pair:
        pairs.append(''.join(current_pair).strip())
    
    return pairs

def _process_dict_pairs(pairs: List[str]) -> Dict[str, str]:
    """Process dictionary pairs into a clean dictionary."""
    result_dict = {}
    for pair in pairs:
        if ':' in pair:
            key, value = pair.split(':', 1)
            key = key.strip().strip("'\"")
            value = _clean_dict_value(value.strip())
            result_dict[key] = value
    
    return result_dict

def _clean_dict_value(value: str) -> str:
    """Clean and format dictionary values."""
    if (value.startswith("'") and value.endswith("'")) or \
       (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    
    value = value.replace('\\', '\\\\')  # Escape backslashes first
    value = value.replace("'", "\\'")    # Escape single quotes
    value = value.replace('"', '\\"')    # Escape double quotes
    value = value.replace('\n', '\\n')   # Escape newlines
    
    return value 