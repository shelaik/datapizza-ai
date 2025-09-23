# NodeSplitter

<!-- prettier-ignore -->
::: datapizza.modules.splitters.NodeSplitter
    options:
        show_source: false


## Usage

```python
from datapizza.modules.splitters import NodeSplitter

splitter = NodeSplitter(
    max_char=800,
)

node_chunks = splitter.split(document_node)
```

## Features

- Maintains Node object structure and hierarchy
- Preserves metadata from original nodes
- Respects node boundaries when possible
- Supports both structure-preserving and flattened chunking
- Handles nested node relationships intelligently

## Examples

### Basic Node Splitting

```python
from datapizza.modules.parsers import TextParser
from datapizza.modules.splitters import NodeSplitter

# Parse text into nodes
parser = TextParser()
document = parser.parse("""
This is the first section of the document.
It contains important information about the topic.

This is the second section with more details.
It provides additional context and examples.

The final section concludes the document.
It summarizes the key points discussed.
""")

splitter = NodeSplitter(
    max_char=150,
)

chunks = splitter.split(document)

# Examine the structured chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:")
    print(f"  Type: {chunk.node_type.value}")
    print(f"  Content length: {len(chunk.content)}")
    print(f"  Children: {len(chunk.children)}")
    print(f"  Content preview: {chunk.content[:80]}...")
    print("---")
```

