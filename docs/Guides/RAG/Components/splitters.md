# Splitters

Splitters are pipeline components that break down documents and nodes into smaller, manageable chunks for processing. The `datapizza-ai` library provides several specialized splitter implementations.

## Available Splitters

### NodeSplitter

A hierarchical splitter that processes Node objects based on character limits.

```python
from datapizza.modules.splitters import NodeSplitter

splitter = NodeSplitter(max_char=5000)
chunks = splitter(node)
```

**Parameters:**
 
- `max_char` (int): Maximum characters per chunk (default: 5000)

**Features:**

- Recursively processes node children when content exceeds limit
- Converts nodes to Chunk objects
- Preserves metadata from original nodes

### RecursiveSplitter

A recursive text splitter with overlap support for maintaining context across chunks.

```python
from datapizza.modules.splitters import RecursiveSplitter

splitter = RecursiveSplitter(max_char=5000, overlap=200)
chunks = splitter(node)
```

**Parameters:**

- `max_char` (int): Maximum characters per chunk (default: 5000)
- `overlap` (int): Character overlap between consecutive chunks (default: 0)

**Features:**

- Collects leaf nodes and groups them into chunks
- Maintains context with configurable overlap
- Merges bounding regions metadata
- Handles oversized nodes by creating dedicated chunks


### TextSplitter

A simple text splitter that splits text into chunks of a specified maximum length with optional overlap.

```python

text_splitter = TextSplitter(max_char=15, overlap=2)
chunks = text_splitter.run("Lorem impsum dolor sit amet")

print(chunks)
# [
# Chunk(id='1', text='Lorem impsum do', embeddings=[], metadata={}),
# Chunk(id='2', text='dolor sit amet', embeddings=[], metadata={}),
# ]

```

### PDFImageSplitter

Converts PDF pages to individual images for visual processing.

```python
from datapizza.modules.splitters import PDFImageSplitter

splitter = PDFImageSplitter(
    image_format="png",
    output_base_dir="output_images",
    dpi=300
)
chunks = splitter("document.pdf")
```

**Parameters:**

- `image_format` (str): Output format - 'png' or 'jpeg' (default: 'png')
- `output_base_dir` (str|Path): Directory for output images (default: 'output_images')
- `dpi` (int): Image resolution (default: 300)

**Features:**

- Uses PyMuPDF (fitz) for PDF processing
- Creates unique subdirectories per PDF
- Generates UUID-based filenames
- Returns Chunk objects with image path metadata

**Dependencies:**

- Requires `PyMuPDF` package (`pip install PyMuPDF`)


## Usage Patterns

### Node splitter
```python
from datapizza.modules.splitters import NodeSplitter
from datapizza.type import Node
from dotenv import load_dotenv

document_node1 = Node(content="This is a very long text that should be split into chunks")
document_node2 = Node(content="This is a second very long text that should be split into chunks")
document = Node(children=[document_node1, document_node2])

splitter = NodeSplitter(max_char=10, overlap=10)
chunks = splitter.run(document)

print(chunks)
# [
# Chunk(id='4c36910f-fcfa-457b-a111-9e961b3767ad', text='This is a very long text that should be split into chunks', embeddings=[], metadata={'boundingRegions': []}), 
# Chunk(id='3661ddc3-3e99-40d3-b939-1b60fac6d932', text='This is a second very long text that should be split into chunks', embeddings=[], metadata={'boundingRegions': []})
#]
```

### Recursive splitter
```python
from datapizza.modules.splitters import RecursiveSplitter
from datapizza.type import Node
from dotenv import load_dotenv

document_node1 = Node(content="This is a very long text that should be split into chunks")
document_node2 = Node(content="This is a second very long text that should be split into chunks")
document = Node(children=[document_node1, document_node2])

splitter = RecursiveSplitter(max_char=10, overlap=10)
chunks = splitter.run(document)

print(chunks)
# [
# Chunk(id='4c36910f-fcfa-457b-a111-9e961b3767ad', text='This is a very long text that should be split into chunks', embeddings=[], metadata={'boundingRegions': []}), 
# Chunk(id='3661ddc3-3e99-40d3-b939-1b60fac6d932', text='This is a second very long text that should be split into chunks', embeddings=[], metadata={'boundingRegions': []})
#]
```
