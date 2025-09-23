# RecursiveSplitter


<!-- prettier-ignore -->
::: datapizza.modules.splitters.RecursiveSplitter
    options:
        show_source: false




## Usage

```python
from datapizza.modules.splitters import RecursiveSplitter
from datapizza.modules.parsers import TextParser

splitter = RecursiveSplitter(
    max_char=1000,
    overlap=100,
)

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



chunks = splitter.split(document)
```

## Features

- Uses multiple separator strategies in order of preference
- Recursive approach ensures optimal chunk boundaries
- Configurable chunk size and overlap for context preservation
- Handles various content types with appropriate separator selection
- Preserves content structure while maintaining size limits
