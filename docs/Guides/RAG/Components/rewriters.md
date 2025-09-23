# Rewriters

Rewriters are pipeline components that transform and enhance user queries using language models and tools. They help optimize queries for better search results and data retrieval by rephrasing, expanding, or restructuring the input.

## Available Rewriters

### ToolRewriter

A tool-based query rewriter that uses LLMs to transform user queries through structured tool interactions.

```python
import os

from datapizza.clients.openai_client import OpenAIClient
from datapizza.modules.rewriters import ToolRewriter
from dotenv import load_dotenv

load_dotenv()

client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
rewriter = ToolRewriter(
    client=client,
    system_prompt="rewrite the query to perform a better search in a vector database",
)

enhanced_query = rewriter.run(user_prompt = "hi, how are u?, explain recursion in python?", memory=None)
print(enhanced_query)
# recursion in Python explanation and examples
```

**Parameters:**

- `client` (Client): LLM client for processing queries
- `system_prompt` (str, optional): Instructions for query transformation
- `tool` (Tool, optional): Custom tool for query processing (defaults to built-in search tool)
- `tool_choice` (str): Tool usage strategy - "required", "auto", or "none" (default: "required")
- `tool_output_name` (str): Name of the tool output parameter (default: "query")

**Features:**

- Uses structured tool calls for consistent query transformation
- Supports custom tools for specialized rewriting logic
- Provides both sync and async execution methods
- Integrates with memory systems for context-aware rewriting
- Built-in vectorstore search tool for database querying

## Default Tool Functionality

`ToolRewriter` uses a tool named `query_database` to trick the LLM into thinking it's making a call to a vector database, to improve accuracy.

```python
@tool
def search_vectorstore(self, query: str) -> list[Chunk]:
    """Search the vectorstore for the most relevant chunks"""
    pass
```

This tool can be overridden with custom implementations for specific use cases.

## Usage Patterns

### Memory-Aware Rewriting
```python
import os

from datapizza.clients.openai_client import OpenAIClient
from datapizza.memory import Memory
from datapizza.modules.rewriters import ToolRewriter
from datapizza.type import ROLE, TextBlock
from dotenv import load_dotenv

load_dotenv()

client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
rewriter = ToolRewriter(
    client=client,
    system_prompt="rewrite the query to perform a better search in a vector database",
)

memory = Memory()
memory.add_turn(blocks=[TextBlock(content="Previous searches were about deep learning")], role=ROLE.USER)

enhanced_query = rewriter.run(
    user_prompt = "and what about nlp?", 
    memory=memory
)
print(enhanced_query)
# natural language processing and deep learning
```

### Async Processing
```python
enhanced_query = await rewriter.a_run("search query", memory=None)
```


## Multi-Step Query Processing
```python
import os

from datapizza.clients.openai_client import OpenAIClient
from datapizza.modules.rewriters import ToolRewriter
from dotenv import load_dotenv

load_dotenv()

client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")

domain_rewriter = ToolRewriter(
    client=client,
    system_prompt="Add domain-specific terminology"
)

clarity_rewriter = ToolRewriter(
    client=client,
    system_prompt="Make queries more specific and clear"
)

step1 = domain_rewriter.run( user_prompt = "nlp")
final_query = clarity_rewriter.run(user_prompt = step1)
print(final_query)
# natural language processing applications and technologies
```
