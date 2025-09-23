# Parsers

Parsers are pipeline components that convert documents into structured hierarchical Node representations. They extract text, layout information, and metadata from various document formats to create tree-like data structures for further processing.

Each parser should return a [Node](../Other_Concepts/node.md) object, which is a hierarchical representation of the document content.

If you write a custom parser that returns a different type of object (for example, the plain text of the document content), you must use a [TreeBuilder](./llm_treebuilder.md) to convert it into a Node.

## Available Parsers

### AzureParser

A document parser that uses Azure AI Document Intelligence to extract structured content from PDFs and other documents.

```python
from datapizza.modules.parsers import AzureParser

parser = AzureParser(
    api_key="your-azure-key",
    endpoint="https://your-endpoint.cognitiveservices.azure.com/",
    result_type="text"
)

document_node = parser("document.pdf")
```

**Parameters:**

- `api_key` (str): Azure AI Document Intelligence API key
- `endpoint` (str): Azure service endpoint URL
- `result_type` (str): Output format - "text" or "markdown" (default: "text")

**Features:**

- Creates hierarchical document structure: document → sections → paragraphs/tables/figures
- Extracts bounding regions and spatial layout information
- Handles tables, figures, and complex document layouts
- Preserves metadata including page numbers and coordinates
- Supports both sync and async processing
- Converts media elements to base64 images with coordinates
- Automatically handles missing paragraphs in document structure

**Node Types Created:**

- `DOCUMENT`: Root document container
- `SECTION`: Document sections
- `PARAGRAPH`: Text paragraphs with content
- `TABLE`: Tables with markdown representation
- `FIGURE`: Images and figures with media data

### TextParser

A simple text parser that creates hierarchical structures from plain text.

```python
from datapizza.modules.parsers.text_parser import TextParser, parse_text

# Using the class
parser = TextParser()
document_node = parser.parse("Your text content here", metadata={"source": "example"})

# Using the convenience function
document_node = parse_text("Your text content here")
```

**Features:**

- Splits text into paragraphs based on double newlines
- Breaks paragraphs into sentences using regex patterns
- Creates three-level hierarchy: document → paragraphs → sentences
- Preserves original text content in sentence nodes
- Adds index metadata for paragraphs and sentences

**Node Types Created:**

- `DOCUMENT`: Root document container
- `PARAGRAPH`: Text paragraphs
- `SENTENCE`: Individual sentences with content

## Usage Patterns

### Azure Document Processing
```python
from datapizza.modules.parsers import AzureParser
import os

parser = AzureParser(
    api_key=os.getenv("AZURE_DOC_INTELLIGENCE_KEY"),
    endpoint=os.getenv("AZURE_DOC_INTELLIGENCE_ENDPOINT"),
    result_type="markdown"
)

# Parse document
document = parser("complex_document.pdf")

# Access hierarchical structure
for section in document.children:
    for paragraph in section.children:
        print(f"Content: {paragraph.content}")
        print(f"Bounding regions: {paragraph.metadata.get('boundingRegions', [])}")
```

### Async Processing
```python
async def process_document():
    document = await parser.a_run("document.pdf")
    return document
```

### Text Processing Pipeline
```python
from datapizza.modules.parsers.text_parser import parse_text

# Process plain text
text_content = """
This is the first paragraph.
It contains multiple sentences.

This is the second paragraph.
It also has content.
"""

document = parse_text(text_content, metadata={"source": "user_input"})

# Navigate structure
for i, paragraph in enumerate(document.children):
    print(f"Paragraph {i}:")
    for j, sentence in enumerate(paragraph.children):
        print(f"  Sentence {j}: {sentence.content}")
```
