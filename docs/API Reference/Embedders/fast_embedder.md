# FastEmbedder


```python
pip install datapizza-ai-embedders-fastembedder
```

<!-- prettier-ignore -->
::: datapizza.embedders.fastembedder.FastEmbedder
    options:
        show_source: false


## Usage

```python
from datapizza.embedders.fastembedder import FastEmbedder

embedder = FastEmbedder(
    model_name="BAAI/bge-base-en-v1.5",
    embedding_name="bge_embeddings",  # Optional custom name
    cache_dir="/path/to/cache"        # Optional cache directory
)

# Embed text (returns sparse embeddings)
embeddings = embedder.embed("Hello world")
```

## Features

- Uses FastEmbed for efficient sparse text embeddings
- Local model execution (no API calls required)
- Configurable model caching directory
- Custom embedding naming
- Sparse embedding format for memory efficiency
- Both sync and async embedding support

## Examples

### Basic Sparse Text Embedding

```python
from datapizza.embedders.fastembedder import FastEmbedder

embedder = FastEmbedder(
    model_name="BAAI/bge-base-en-v1.5",
    embedding_name="my_embeddings"
)

# Single text embedding
text = "This is a sample document for sparse embedding."
result = embedder.embed(text)

print(f"Embedding name: {list(result.keys())[0]}")
print(f"Sparse embedding format: {type(result['my_embeddings'])}")
```

### Custom Cache Directory

```python
from datapizza.embedders.fastembedder import FastEmbedder
import os

# Use custom cache directory for model storage
cache_path = os.path.expanduser("~/my_models_cache")

embedder = FastEmbedder(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir=cache_path
)

text = "Text to embed with custom cache location."
embeddings = embedder.embed(text)
```

### Different Model Selection

```python
from datapizza.embedders.fastembedder import FastEmbedder

# Use different sparse embedding models
models = [
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/all-MiniLM-L6-v2"
]

for model in models:
    embedder = FastEmbedder(
        model_name=model,
        embedding_name=f"{model.split('/')[-1]}_embeddings"
    )

    result = embedder.embed("Sample text for comparison")
    embedding_name = list(result.keys())[0]
    print(f"Model: {model}")
    print(f"Embedding name: {embedding_name}")
    print("---")
```

### Async Embedding

```python
import asyncio
from datapizza.embedders.fastembedder import FastEmbedder

async def embed_async():
    embedder = FastEmbedder(model_name="BAAI/bge-base-en-v1.5")

    text = "Async sparse embedding example"
    result = await embedder.a_embed(text)

    return result

# Run async function
result = asyncio.run(embed_async())
```