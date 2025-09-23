# Clients

This guide provides an overview of the available clients in `datapizza-ai` for interacting with different AI providers.


```python
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(
    api_key="YOUR_APY_KEY",
    model="gpt-4o-mini",
)
# Basic text response
response = client.invoke("Hello!")
print(response.text)
```



All clients extend the base `Client` class and provide consistent interfaces for:

- Single inference requests
- Streaming responses
- Structured output (where supported)
- Tool/function calling capabilities
- Conversation [memory](../Other_Concepts/memory.md) management

## Base Client Interface

All clients implement these core methods:

```python
def invoke(self, input: str | List[Block], tools: List[str] = [], memory: Memory = None, tool_choice: str = "auto", ...) -> ClientResponse
def a_invoke(self, input: str | List[Block], tools: List[str] = [], memory: Memory = None, tool_choice: str = "auto", ...) -> Awaitable[ClientResponse]
def stream_invoke(self, input: str | List[Block], tools: List[str] = [], memory: Memory = None, tool_choice: str = "auto", ...) -> Iterator[ClientResponse]
def a_stream_invoke(self, input: str | List[Block], tools: List[str] = [], memory: Memory = None, tool_choice: str = "auto", ...) -> Iterator[ClientResponse]
def structured_response(self, input: str, output_cls: Type[Model], memory: Memory = None, ...) -> ClientResponse
def a_structured_response(self, input: str, output_cls: Type[Model], memory: Memory = None, ...) -> ClientResponse
def embed(self, text: Union[str, list[str]], model_name: str = None) -> List[float]
def a_embed(self, text: Union[str, list[str]], model_name: str = None) -> List[float]
```

## Common Parameters

All clients support these common parameters:

- `input`: The input text/prompt to send to the model (can be string or List[Block])
- `tools`: List of [tools](../../API%20Reference/Type/tool.md) available for the model to use
- `memory`: [Memory](../Other_Concepts/memory.md)) object containing conversation history
- `tool_choice`: Controls which tool to use ("auto" by default)
- `temperature`: Controls randomness in responses (0.0 to 1.0)
- `max_tokens`: Maximum number of tokens in the response
- `system_prompt`: System-level instructions for the model
- `cache`: A cache instance


## Instantiate 

```python
from datapizza.clients.openai import OpenAIClient

client = OpenAIClient(
    api_key="YOUR_APY_KEY",
    model="gpt-4o-mini",
    system_prompt="You are a helpful assistant",
    temperature=0.5,
)
```

## Basic Invocation
```python
# With temperature control
response = client.invoke(
    input="What is the weather like?",
    temperature=0.7
)
print(response.text)
```


## Multimodality

| Client     | Image (URL/base64/path) | PDF (base64/path/raw)  | Audio (path/raw) |
|------------|-------------------------|------------------------|------------------|
| OpenAI     | ✓                       | ✓                      |                  |
| Google     | ✓                       | ✓                      | ✓                |
| Anthropic  | ✓                       | ✓                      |                  |
| Mistral    | ✓                       |                        |                  |

\* OpenAI `gpt-4o-audio-preview` and `gpt-4o-mini-audio-preview` models supports audio input



```python
from datapizza.type import Media, MediaBlock, TextBlock

img1 = Media(
    media_type="image",
    source_type="path",
    source="gogole.png",
    extension="png",
)
media_block1 = MediaBlock(media=img1)

img2 = Media(
    media_type="image",
    source_type="path",
    source="microsoft.png",
    extension="png",
)
media_block2 = MediaBlock(media=img2)

text_block = TextBlock(content="cosa vedi in queste immagini?")

response = client.invoke(input=[text_block, media_block1, media_block2],  max_tokens=100)

print(response.text)
```

## Using Tools
```python
from datapizza.tools import tool

@tool
def timer_tool(location:str)
    pass # DO something

response = client.invoke(
    "Set a timer for 5 minutes",
    tools=[timer_tool]
)

for f_call in response.function_calls:
    f_call(**f_call.arguments)
```

## Structured Responses
```python
from pydantic import BaseModel

class UserSchema(BaseModel):
    username: str
    email: str

response = client.structured_response(
    input="Get user info",
    output_cls=UserSchema
)
user_data = response.structured_data[0]
print(f"Username: {user_data.username}")
```


## Streaming Responses
```python
for chunk in client.stream_invoke("Long response..."):
    if chunk.delta:
        print(chunk.delta, end="")
```

## Async methods
### Async Invoke
```python
import asyncio

async def get_response():
    response = await client.a_invoke(
        input="What are the main AI research trends in 2024?",
        temperature=0.7
    )
    return response.text

asyncio.run(get_response())
```

### Async streaming
The last iteration will return a ClientResponse with token usage
```python
import asyncio

async def get_response():
    async for chunk in client.a_stream_invoke("tell me a joke"):
        print(chunk.delta)

asyncio.run(get_response())
```

### Async structured Responses

```python
class UserSchema(BaseModel):
    username: str
    email: str

async def get_response():
    response = client.a_structured_response(
        input="Get user info",
        output_cls=UserSchema
    )
    user_data = response.structured_data[0]
    print(f"Username: {user_data.username}")

asyncio.run(get_response())

```

## Embeddings
```python
# Embed a single text
embeddings = client.embed("This is a sample text")

# Embed multiple texts
embeddings = client.embed(["Text 1", "Text 2", "Text 3"])
```

## Real-World Example

Here's a simple example of a chatbot

```python
from datapizza.clients.openai_client import OpenAIClient
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock

client = OpenAIClient(api_key="API_KEY", model="gpt-4o")

memory = Memory()
user_question = input("Enter your question (type 'exit' to end): ")

while user_question != "exit":
    response = client.invoke(user_question, memory=memory)
    print("LLM response: ", response.text)

    memory.add_turn(TextBlock(content=user_question), role=ROLE.USER)
    memory.add_turn(response.content, role=ROLE.ASSISTANT)
    user_question = input("Enter your question (type 'exit' to end): ")

```
