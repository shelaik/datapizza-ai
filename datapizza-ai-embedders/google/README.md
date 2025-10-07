# Google Embedder

Google Generative AI embedder implementation for datapizza-ai

## Installation

```bash
pip install datapizza-ai-embedders-google
```

## Usage

```python
from datapizza.embedders.google import GoogleEmbedder

embedder = GoogleEmbedder(api_key="your-google-api-key")
embeddings = embedder.embed("Hello world", model_name="models/text-embedding-004")
```
