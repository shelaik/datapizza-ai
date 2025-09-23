# TextSplitter


<!-- prettier-ignore -->
::: datapizza.modules.splitters.TextSplitter
    options:
        show_source: false

## Usage

```python
from datapizza.modules.splitters import TextSplitter

splitter = TextSplitter(
    max_char=500,
    overlap=50
)

chunks = splitter.split(text_content)
```

## Features

- Simple, straightforward text splitting algorithm
- Configurable chunk size and overlap
- Lightweight implementation for basic splitting needs
- Preserves character-level accuracy in chunk boundaries
- Minimal overhead for high-performance applications

## Examples

### Basic Usage

```python
from datapizza.modules.splitters import TextSplitter

splitter = TextSplitter.split(max_char=500, overlap=50)

text = """
This is a sample text that we want to split into smaller chunks.
The TextSplitter will divide this content based on the specified
chunk size and overlap parameters. This ensures that information
is preserved while creating manageable pieces of content.
"""

chunks = splitter(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {len(chunk.content)} chars")
    print(f"Content: {chunk.content}")
    print("---")
```
