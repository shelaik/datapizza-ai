# Rewriters

Rewriters are pipeline components that transform and rewrite text content using language models. They can modify content style, format, tone, or structure while preserving meaning and important information.

<!-- prettier-ignore -->
::: datapizza.modules.rewriters.ToolRewriter
    options:
        show_source: false




A rewriter that uses language models to transform text content with specific instructions and tools.

```python
from datapizza.modules.rewriters import ToolRewriter
from datapizza.clients import OpenAIClient

client = OpenAIClient(api_key="your-api-key")
rewriter = ToolRewriter(
    client=client,
    system_prompt="You are an expert content rewriter.",
    rewrite_instructions="Rewrite the following text to be more concise and clear."
)

# Rewrite text content
original_text = "This is a lengthy and complex explanation of a simple concept."
rewritten_text = rewriter.rewrite(original_text)
```

**Features:**

- Flexible content transformation with custom instructions
- Support for various rewriting tasks (summarization, style changes, format conversion)
- Integration with tool calling for enhanced capabilities
- Preserves important information while transforming presentation
- Supports both sync and async processing
- Configurable prompting for different rewriting strategies

## Usage Examples

### Basic Content Rewriting
```python
from datapizza.modules.rewriters import ToolRewriter
from datapizza.clients import OpenAIClient

client = OpenAIClient(api_key="your-openai-key")

# Create a simplification rewriter
simplifier = ToolRewriter(
    client=client,
    system_prompt="You are an expert at simplifying complex text for general audiences.",
    rewrite_instructions="Rewrite the following text to be simpler and more accessible, using everyday language."
)

# Simplify technical content
technical_text = """
The algorithmic implementation utilizes a recursive binary search methodology
to optimize computational complexity in logarithmic time scenarios.
"""

simplified_text = simplifier(technical_text)
print(simplified_text)
# Output: "This method uses a smart search technique that quickly finds information
# by repeatedly splitting the search area in half."
```
