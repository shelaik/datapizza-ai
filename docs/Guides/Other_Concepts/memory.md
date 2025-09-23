# Memory

This guide explains the memory system in `datapizza-ai`, which provides a structured way to store and manage conversation history.

## Overview

The memory system consists of two main classes:

1. `Turn` - Represents a single conversation turn with a role (assistant or user) and a collection of blocks
2. `Memory` - Manages the full conversation history as a sequence of turns

## Turn Class

The `Turn` class represents a single conversation turn containing one or more [blocks](./block.md) and a role (assistant, user, system, or tool).

```python
turn = Turn(blocks=[text_block, image_block], role=ROLE.ASSISTANT)
```

### Properties

- `blocks`: List of Block objects (text, function calls, or structured data)
- `role`: ROLE enum value (`ASSISTANT`, `USER`, `SYSTEM` or `TOOL`)

### Methods

- `append(block)`: Add a block to the turn
- `extend(blocks)`: Add multiple blocks
- `insert(index, block)`: Insert a block at a specific position

## Memory Class

The `Memory` class stores the complete conversation history as a sequence of turns.

```python
memory = Memory()
memory.add_turn(text_block, ROLE.USER)
memory.add_to_last_turn(response_block)
```

### Key Methods

- `new_turn(role)`: Start a new conversation turn
- `add_turn(blocks, role)`: Add a complete turn with blocks
- `add_to_last_turn(block)`: Append a block to the most recent turn
- `clear()`: Reset the memory
- `iter_blocks()`: Iterate through all blocks in all turns
- `json_dumps()`: Serialize the memory to JSON
- `json_loads()`: Deserialize JSON to the memory

### Utility Features

The Memory class implements several Python magic methods for intuitive usage:

- Iteration (`__iter__`): Iterate through turns
- Indexing (`__getitem__`, `__setitem__`, `__delitem__`): Access turns by index
- Comparison (`__eq__`): Compare memories efficiently using content hashing
- Hashing (`__hash__`): Generate a deterministic hash based on memory content

## Usage Example

```python
from datapizza.type import ROLE
from datapizza.memory.memory import Memory, Turn
from your_block_module import TextBlock

# Create a new memory instance
memory = Memory()

# Add a user message
memory.add_turn(TextBlock("What's the weather today?"), ROLE.USER)

# Add assistant response
memory.add_turn(TextBlock("It's sunny with a high of 75°F."), ROLE.ASSISTANT)

# Add a follow-up in the same turn
memory.add_to_last_turn(TextBlock("Would you like a forecast for tomorrow?"))

# Access turns
user_message = memory[0]
assistant_response = memory[1]

# Iterate through all blocks
for block in memory.iter_blocks():
    print(block)
```


## Real-World Example

Here's a simple example of a chatbot

```python
from datapizza.clients.openai_client import OpenAIClient
from datapizza.memory import Memory
from datapizza.type import ROLE, TextBlock
from dotenv import load_dotenv

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
