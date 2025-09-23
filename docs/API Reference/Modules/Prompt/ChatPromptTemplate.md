# ChatPromptTemplate

The ChatPromptTemplate class provides utilities for managing prompts, prompt templates, and conversation memory for AI interactions. It helps structure and format prompts for various AI tasks and maintain conversation context.

<!-- prettier-ignore -->
::: datapizza.modules.prompt.ChatPromptTemplate
    options:
        show_source: false

## Overview

The ChatPromptTemplate module provides utilities for managing prompts and prompt templates in AI pipelines.

```python
from datapizza.modules.prompt import prompt
```

**Features:**

- Prompt template management and formatting
- Context-aware prompt construction
- Integration with memory systems for conversation history
- Structured prompt formatting for different AI tasks

## Usage Examples

### Basic Prompt Management
```python
from datapizza.modules.prompt import prompt

# Create structured prompts for different tasks
system_prompt = prompt.create_system_prompt(
    role="helpful assistant",
    context="You are helping with data analysis tasks.",
    guidelines=["Be precise", "Show your work", "Use examples"]
)

user_prompt = prompt.create_user_prompt(
    task="Analyze the following data",
    context=data_context,
    examples=example_analyses
)
```
