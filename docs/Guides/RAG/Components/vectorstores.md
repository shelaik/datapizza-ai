# Vectorstores

Vectorstores provide persistent storage and similarity search for embedded document chunks. They enable efficient vector-based retrieval for RAG applications.

### QdrantVectorstore

High-performance vector database implementation using Qdrant for storing and searching embeddings.

```python
from datapizza.vectorstores import QdrantVectorstore

vectorstore = QdrantVectorstore(
    host="localhost",
    port=6333,
    api_key="your-qdrant-key"
)

# Store embeddings
vectorstore.add(embedded_chunks, collection_name="documents")

# Search for similar content
results = vectorstore.search(
    query_vector=query_vector,
    collection_name="documents",
    k=10
)
```

**Parameters:**

- `host` (str): Qdrant server hostname
- `port` (int): Qdrant server port (default: 6333)
- `api_key` (str, optional): Authentication key for Qdrant Cloud

**Features:**

- Dense and sparse vector support
- Batch operations for efficient storage
- Configurable similarity search
- Async operations support
- Metadata filtering capabilities


## Create collection


Create collection if collection does not exist, otherwise it prints a warning.

```python
vectorstore = QdrantVectorstore(location=":memory:")
vectorstore.create_collection(collection_name="knowledge_base", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])
```

## Add documents to the collection


```python
import uuid
from datapizza.clients.openai_client import OpenAIClient
from datapizza.embedders import NodeEmbedder
from datapizza.type import Chunk
from datapizza.vectorstores import QdrantVectorstore

vectorstore = QdrantVectorstore(location=":memory:")

# Setup embedding pipeline
client = OpenAIClient(api_key="your-key", model="text-embedding-3-small")
embedder = NodeEmbedder(client=client)
vectorstore = QdrantVectorstore(host="localhost", port=6333)

# Create and embed chunks
chunks = [
    Chunk(id=uuid.uuid4(), text="Python programming concepts"),
    Chunk(id=uuid.uuid4(), text="Machine learning fundamentals")
]

embedded_chunks = embedder(chunks)

vectorstore.add(embedded_chunks, collection_name="knowledge_base")
```

## Search from the collection

```python
from datapizza.vectorstores import QdrantVectorstore
from datapizza.embedders import NodeEmbedder
from datapizza.clients.openai_client import OpenAIClient
from datapizza.type import Chunk


vectorstore = QdrantVectorstore(location=":memory:")
client = OpenAIClient(api_key="your-key", model="text-embedding-3-small")
query_embedding = client.embed("programming languages")
results = vectorstore.search(
    query_vector=query_embedding,
    collection_name="knowledge_base",
)

print(f"Found {len(results)} similar documents")
# Found 2 similar documents
```
