# Rerankers

Rerankers are pipeline components that reorder search results based on semantic relevance to a query. They improve retrieval quality by using specialized models to score and rank document chunks.

### CohereReranker

Uses Cohere's reranking models to reorder documents by relevance.

```python
from datapizza.modules.rerankers import CohereReranker

reranker = CohereReranker(
    api_key="your-cohere-key",
    endpoint="https://api.cohere.ai/v1",
    top_n=10,
    threshold=0.5
)

reranked_chunks = reranker.run(query="machine learning", documents=chunks)
```

**Parameters:**

- `api_key` (str): Cohere API key
- `endpoint` (str): Cohere API endpoint URL
- `top_n` (int): Maximum number of documents to return (default: 10)
- `threshold` (float, optional): Minimum relevance score threshold

### TogetherReranker

Uses Together AI's reranking models for document reordering.

```python
from datapizza.modules.rerankers import TogetherReranker

reranker = TogetherReranker(
    api_key="your-together-key",
    model="BAAI/bge-reranker-large",
    top_n=10,
    threshold=0.7
)

reranked_chunks = reranker.run(query="neural networks", documents=chunks)
```

**Parameters:**

- `api_key` (str): Together AI API key
- `model` (str): Reranking model name
- `top_n` (int): Maximum number of documents to return (default: 10)
- `threshold` (float, optional): Minimum relevance score threshold

## Features

**Both rerankers support:**

- Relevance-based document ranking
- Configurable result limits (`top_n`)
- Optional score thresholding for quality filtering
- Sync and async processing (CohereReranker only for async)

**Filtering Behavior:**

- Without threshold: Returns top N documents in relevance order
- With threshold: Returns only documents above the score threshold

## Usage Patterns

### Basic Reranking
```python
from datapizza.modules.rerankers import TogetherReranker

reranker = TogetherReranker(
    api_key="your-key",
    model="BAAI/bge-reranker-large",
    top_n=5
)

# Rerank search results
query = "deep learning applications"
reranked = reranker.run(query, search_results)
```

### Quality Filtering
```python
# Only return highly relevant documents
reranker = CohereReranker(
    api_key="your-key",
    endpoint="https://api.cohere.ai/v1",
    threshold=0.8  # High relevance threshold
)

filtered_results = reranker.run(query, candidates)
```

### Async Processing
```python
# CohereReranker supports async
async def rerank_documents():
    results = await reranker.a_run(query, documents)
    return results
```

 
## RAG Pipeline Integration
```python
from datapizza.vectorstores import QdrantVectorstore
from datapizza.modules.rerankers import TogetherReranker

# Initial vector search
vectorstore = QdrantVectorstore(collection_name="docs")
initial_results = vectorstore.search(query, top_k=50)

# Rerank for precision
reranker = TogetherReranker(
    api_key="your-key",
    model="BAAI/bge-reranker-large",
    top_n=10
)

final_results = reranker.run(query, initial_results)
```


