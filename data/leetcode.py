# Preparation of LeetCode dataset
import re

def extract_function_info(text):
    # Regular expression pattern to match the function definition within the code block
    function_pattern = re.compile(r'```python\s*(def\s+(\w+)\s*(\(.*?\)):\s*(.*?))\s*```', re.DOTALL)
    
    # Extract the function information
    function_match = function_pattern.search(text)
    if function_match:
        function_def = function_match.group(1)
        function_name = function_match.group(2)
        function_header = f"def {function_name}{function_match.group(3)}:"
        function_content = function_match.group(4).strip()
    else:
        function_def = None
        function_name = None
        function_header = None
        function_content = None
    
    return function_def, function_name, function_header, function_content