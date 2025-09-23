# Client Response

The `ClientResponse` class is a standardized way to handle responses from different AI providers in `datapizza-ai`. It provides a unified interface for accessing response content, whether it's text, function calls, or structured data.

## Response Structure

A `ClientResponse` contains:

- A list of content [blocks](../Other_Concepts/block.md) (text, function calls, or structured data)
- Optional delta content for streaming responses
- Stop reason from the AI provider
- Token usage metrics (when implemented)

## Content Block Types

### Text Blocks
Simple text responses from the model.

```python
response = client.invoke("Tell me a joke")
text = response.text  # Gets all text concatenated
```

### Function Call Blocks
Represents tool/function calls made by the model.

```python
response = client.invoke("What's the weather?", tools=[WeatherTool()])
function_calls = response.function_calls
for call in function_calls:
    print(f"Function: {call.name}")
    print(f"Arguments: {call.arguments}")
```

### Structured Blocks
Contains structured data (Pydantic models or JSON).

```python
class WeatherInfo(BaseModel):
    temperature: float
    condition: str

response = client.structured_response(
    input="Weather status?",
    output_cls=WeatherInfo
)
data = response.structured_data[0]
```

## Helper Methods

Check response type:
```python
if response.is_pure_text():
    print("Response contains only text")
    
if response.is_pure_function_call():
    print("Response contains only function calls")
```

## Streaming Responses

When using streaming, each chunk is a `ClientResponse` with a delta:

```python
for chunk in client.stream_invoke("Long response..."):
    if chunk.delta:
        print(chunk.delta, end="")
```

## Properties

- `content`: List[Block] - All response blocks in order
- `delta`: str | None - Incremental content for streaming
- `stop_reason`: str | None - Why the model stopped generating
- `text`: str - All text blocks concatenated
- `first_text`: str - Content of first text block
- `function_calls`: List[FunctionCallBlock] - All function calls
- `structured_data`: List[Model] - All structured data blocks
``
