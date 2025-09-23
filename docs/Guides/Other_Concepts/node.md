# Node

A Node is a fundamental building block in `datapizza-ai`'s' document processing system. It represents a hierarchical component of a document in a tree-like structure.

## Overview

The `Node` class represents elements in a document graph, allowing for hierarchical organization of document content. Nodes can represent various document elements such as sections, paragraphs, sentences, pages, tables, or figures.

```python
class Node:
    def __init__(
        self,
        children: list["Node"] | None = None,
        metadata: dict | None = None,
        node_type: NodeType = NodeType.SECTION,
        content: str | None = None,
    ):
        # ...
```

## Node Types

Nodes can represent different elements within a document:

- `SECTION`: A document section
- `PARAGRAPH`: A paragraph of text
- `DOCUMENT`: A complete document
- `SENTENCE`: A single sentence
- `PAGE`: A document page
- `TABLE`: A table structure
- `FIGURE`: An image or diagram

## Node Properties

- `children`: List of child nodes
- `metadata`: Dictionary for storing arbitrary metadata
- `node_type`: The type of the node (from NodeType enum)
- `content`: Text content (for leaf nodes)
- `id`: Unique identifier

## Nodes in the Ingestion Pipeline

Nodes play a critical role in the document ingestion pipeline:

1. **Parsing Stage**: Document parsers convert raw documents into a hierarchical Node structure
     - The parser analyzes document structure and creates appropriate Node types
     - Parent-child relationships preserve document hierarchy
     - Metadata is attached to Nodes for additional context

2. **Processing Stage**: Nodes can be transformed, enhanced, or filtered
     - Additional metadata can be added
     - Content can be processed or transformed
     - Nodes can be reorganized as needed

3. **Splitting Stage**: The Node structure is split into Chunks
     - Splitters break down Nodes into appropriately sized Chunks
     - The hierarchical structure helps create logical splits
     - Metadata from parent Nodes is typically passed to resulting Chunks

4. **Embedding Stage**: Chunks are embedded and stored
     - Each Chunk is processed to create vector embeddings
     - Chunk metadata preserves context from the original Nodes
     - Embedded Chunks are stored in a vector database for retrieval

## Example Flow

```
Document → Parser → Node Tree → Processor → Enhanced Nodes → Splitter → Chunks → Embedder → Vector Store
```

## Special Node Types

### MediaNode

For documents containing media elements:

```python
class MediaNode(Node):
    def __init__(
        self,
        media: Media,
        children: list["Node"] | None = None,
        metadata: dict | None = None,
        node_type: NodeType = NodeType.SECTION,
        content: str | None = None,
    ):
        # ...
```

## Related Concepts

### Chunk

The output of the Node splitting process:

```python
@dataclass
class Chunk:
    id: str
    text: str
    embeddings: list[Embedding] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

Chunks are designed to be:

- Appropriately sized for embedding
- Self-contained units of information
- Enriched with metadata from their source Nodes
- Ready for ingestion into vector databases
