# Prompt Templates

Prompt templates create structured conversation memories by combining user queries, retrieved chunks, and function calls using Jinja2 templating.

### ChatPromptTemplate

Creates formatted memory objects with user prompts, retrieval results, and function call patterns for RAG systems.

```python
from datapizza.modules.prompt import ChatPromptTemplate

template = ChatPromptTemplate(
    user_prompt_template="User question: {{ user_prompt }}",
    retrieval_prompt_template="Retrieved content:\n{% for chunk in chunks %}{{ chunk.text }}\n{% endfor %}"
)

memory = template.format(
    user_prompt="What is machine learning?",
    chunks=search_results,
    retrieval_query="machine learning definition"
)
```

**Parameters:**

- `user_prompt_template` (str): Jinja2 template for formatting user input
- `retrieval_prompt_template` (str): Jinja2 template for formatting retrieved chunks

**Features:**

- Jinja2 templating for flexible prompt formatting
- Automatic function call simulation for RAG patterns
- Memory preservation from previous conversations
- Sync and async processing support

## Real-World Example

```python
from datapizza.modules.prompt import ChatPromptTemplate
from datapizza.type import Chunk

# Create RAG prompt template
template = ChatPromptTemplate(
    user_prompt_template="Question: {{ user_prompt }}\nPlease answer based on the provided context.",
    retrieval_prompt_template="Context:\n{% for chunk in chunks %}- {{ chunk.text }}\n{% endfor %}"
)

# Simulate search results
chunks = [
    Chunk(id="1", text="Python is a high-level programming language"),
    Chunk(id="2", text="Python was created by Guido van Rossum in 1991")
]

# Create conversation memory
memory = template.format(
    user_prompt="Who created Python?",
    chunks=chunks,
    retrieval_query="Python creator history"
)

print("User: ", memory[0])
print("Assistant: ", memory[1])
print("Tool: ", memory[2].blocks[0].result)

# 1. User:  Question: Who created Python? Please answer based on the provided context.
# 2. Assistant: FunctionCall(search_vectorstore, query="Python creator history")
# 3. Tool:  Context - Python is a high-level programming language - Python was created by Guido van Rossum in 1991
```
