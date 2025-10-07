# OpenAI Embedder

OpenAI embedder implementation.

## Installation

```bash
pip install datapizza-ai-embedders-openai
```

## Usage

```python
from datapizza.embedders.openai import OpenAIEmbedder

embedder = OpenAIEmbedder(api_key="your-openai-api-key")
embeddings = embedder.embed("Hello world", model_name="text-embedding-ada-002")
```
