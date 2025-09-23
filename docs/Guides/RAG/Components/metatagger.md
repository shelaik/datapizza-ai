# Metatagger

Metatagger components extract and add metadata to document chunks using LLMs. They enhance chunks with structured information for improved search and categorization.

### KeywordMetatagger

Extracts keywords from text chunks using LLM-powered analysis.

```python
from datapizza.modules.metatagger import KeywordMetatagger
from datapizza.clients.openai_client import OpenAIClient

client = OpenAIClient(api_key="your-key", model="gpt-4o-mini")
tagger = KeywordMetatagger(
    client=client,
    max_workers=3,
    system_prompt="Extract 5 relevant keywords from the text",
    keyword_name="keywords"
)

tagged_chunks = tagger(chunks)
```

**Parameters:**

- `client` (Client): LLM client for keyword extraction
- `max_workers` (int): Concurrent processing threads (default: 3)
- `system_prompt` (str, optional): Instructions for keyword extraction
- `user_prompt` (str, optional): Additional user context
- `keyword_name` (str): Metadata field name for keywords (default: "keywords")

**Features:**

- Concurrent chunk processing
- Structured keyword extraction using Pydantic models
- Customizable prompts and metadata field names
- Preserves original chunk content and IDs

## Real-World Example

```python
import os
from datapizza.clients.openai_client import OpenAIClient
from datapizza.modules.metatagger import KeywordMetatagger
from datapizza.type import Chunk
from dotenv import load_dotenv

chunks = [
    Chunk(id="1", text="Machine learning algorithms use statistical methods to identify patterns in data"),
    Chunk(id="2", text="Neural networks consist of interconnected nodes that process information")
]

client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
tagger = KeywordMetatagger(
    client=client,
    system_prompt="Extract technical keywords and concepts from this text"
)

tagged_chunks = tagger(chunks)

# Access keywords in metadata
for chunk in tagged_chunks:
    print(f"Text: {chunk.text[:50]}...")
    print(f"Keywords: {chunk.metadata['keywords']}")

    # Text: Machine learning algorithms use statistical method...
    # Keywords: ['machine learning algorithms', 'statistical methods', 'patterns in data']

    # Text: Neural networks consist of interconnected nodes th...
    # Keywords: ['Neural networks', 'interconnected nodes', 'information processing']
```
