# Embedders

Embedders convert text and images into vector representations for semantic search and similarity matching. They support various embedding models and batch processing.

### ClientEmbedder

Basic text embedder using any compatible LLM client with embedding capabilities.

```python
from datapizza.embedders import ClientEmbedder
from datapizza.clients.openai_client import OpenAIClient

client = OpenAIClient(api_key="your-key")
embedder = ClientEmbedder(
    client=client,
    model_name="text-embedding-3-small"
)

embeddings = embedder.embed(text = "Your text here")
```

### NodeEmbedder

Batch processor for embedding multiple chunks efficiently with configurable batch sizes.

```python
from datapizza.embedders import NodeEmbedder

embedder = NodeEmbedder(
    client=client,
    model_name="text-embedding-3-small",
    batch_size=100
)

embedded_chunks = embedder(chunks)
```

**Parameters:**

- `client` (Client): LLM client with embedding support
- `model_name` (str, optional): Specific embedding model name
- `embedding_name` (str, optional): Custom name for embeddings
- `batch_size` (int): Chunks per batch (NodeEmbedder only, default: 2047)

**Features:**

- Sync and async processing support
- Configurable embedding models
- Batch processing for efficiency
- Compatible with all `datapizza-ai` clients

## Real-World Example

```python
import os

from datapizza.clients.openai_client import OpenAIClient
from datapizza.embedders import NodeEmbedder
from datapizza.type import Chunk
from dotenv import load_dotenv

load_dotenv()


# Embed document chunks for vector search
chunks = [
    Chunk(id="1", text="Python is a programming language"),
    Chunk(id="2", text="Machine learning uses statistical methods")
]

client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"))
embedder = NodeEmbedder(
    client=client,
    model_name="text-embedding-3-small",
    batch_size=50
)

embedded_chunks = embedder(chunks)

# Chunks now have dense embeddings for vector search
for chunk in embedded_chunks:
    print(f"Text: {chunk.text}")
    print(f"Embedding dimensions: {len(chunk.embeddings[0].vector)}")
    # Text: Python is a programming language
    # Embedding dimensions: 1536

```
