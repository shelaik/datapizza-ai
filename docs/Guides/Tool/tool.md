# Tools

## Introduction

The tools module provides a simple way to convert Python functions into tool objects that preserve their behavior while adding metadata like parameter descriptions and types. These tools can be used to create interfaces for your functions that are self-documenting and easier to discover and use.

## Basic Usage

The simplest way to create a tool is to use the `@tool` decorator:

```python
from datapizza.tools import tool

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

This creates a callable tool object that behaves like the original function, but with added metadata.

## Tool Customization

You can customize tools by passing arguments to the decorator:

```python
@tool(
    name="addition", 
    description="Add two numbers and return the sum",
)
def add_numbers(a: int, b: int) -> int:
    return a + b
```

### Parameters

- `name`: Custom name for the tool (defaults to function name)
- `description`: Custom description (defaults to function docstring)
- `end`: Boolean flag indicating if this tool ends a chain of operations

## Tool Metadata

Tools automatically extract metadata from your function:

- Required parameters (those without default values)
- Parameter types from type annotations
- Default values
- Function documentation

This metadata is available through the tool's attributes:

```python
print(add_numbers.name)         # "addition" or "add_numbers" if not specified
print(add_numbers.description)  # The description or docstring
print(add_numbers.properties)   # Dict of parameters and their metadata
print(add_numbers.required)     # List of required parameter names
print(add_numbers.schema)       # Complete schema for the function in openAI standard
```


## Arguments custom description

If you want to change arguments description you can do it 

```py
from typing import Annotated

@tool
def my_func(a : Annotated[int, "this is a description"]):
    return a + 1


print(my_func.schema["parameters"])
#{'a': {'type': 'integer', 'description': 'this is a description'}}
```
This uses `Annotated` from the `typing` module to attach a custom description to the function parameter `a`. This can be helpful in tools or frameworks that use these annotations to generate documentation, validate inputs, or create user interfaces dynamically.


## Advanced Usage: Manual Tool Creation

You can create tools programmatically:

```python
from datapizza.tools import Tool

def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
    
multiply_tool = Tool(
    func=multiply,
    name="multiplication",
    description="Multiply two numbers together"
)

# Use it like a normal function
result = multiply_tool(5, 3)
# result -> 15
```

## Best Practices

1. Always add type annotations to help with parameter metadata
2. Provide clear docstrings to describe tool functionality
3. Use meaningful parameter names
4. Consider setting default values for optional parameters
5. Keep tool functions focused on a single responsibility
