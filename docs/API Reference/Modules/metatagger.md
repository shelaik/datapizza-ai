# Metatagger

Metataggers are pipeline components that add metadata tags to content chunks using language models. They analyze text content and generate relevant keywords, tags, or other metadata to enhance content discoverability and organization.

<!-- prettier-ignore -->
::: datapizza.modules.metatagger.KeywordMetatagger
    options:
        show_source: false



A metatagger that uses language models to generate keywords and metadata for text chunks.

```python
from datapizza.modules.metatagger import KeywordMetatagger
from datapizza.clients import OpenAIClient

client = OpenAIClient(api_key="your-api-key")
metatagger = KeywordMetatagger(
    client=client,
    max_workers=3,
    system_prompt="Generate relevant keywords for the given text.",
    user_prompt="Extract 5-10 keywords from this text:",
    keyword_name="keywords"
)

# Process chunks
tagged_chunks = metatagger.tag(chunks)
```

**Features:**

- Processes chunks in parallel for better performance
- Configurable prompts for different keyword extraction strategies
- Adds generated keywords to chunk metadata
- Supports custom metadata field naming
- Handles both individual chunks and lists of chunks
- Uses memory-based conversation for consistent prompting

**Input/Output:**

- Input: `Chunk` objects or lists of `Chunk` objects
- Output: Same `Chunk` objects with additional metadata containing generated keywords

## Usage Examples

### Basic Keyword Extraction
```python
from datapizza.modules.metatagger import KeywordMetatagger
from datapizza.clients import OpenAIClient
from datapizza.type import Chunk

# Initialize client and metatagger
client = OpenAIClient(api_key="your-openai-key")
metatagger = KeywordMetatagger(
    client=client,
    system_prompt="You are a keyword extraction expert. Generate relevant, concise keywords.",
    user_prompt="Extract 5-8 important keywords from this text:",
    keyword_name="keywords"
)

# Process chunks
chunks = [
    Chunk(content="Machine learning algorithms are transforming healthcare diagnostics."),
    Chunk(content="Climate change impacts ocean temperatures and marine ecosystems.")
]

tagged_chunks = metatagger.tag(chunks)

# Access generated keywords
for chunk in tagged_chunks:
    print(f"Content: {chunk.content}")
    print(f"Keywords: {chunk.metadata.get('keywords', [])}")
```

### Custom Metadata Fields
```python
# Create different metataggers for different types of metadata
topic_tagger = KeywordMetatagger(
    client=client,
    user_prompt="Identify the main topic category for this text:",
    keyword_name="topic"
)

sentiment_tagger = KeywordMetatagger(
    client=client,
    user_prompt="Analyze the sentiment of this text (positive/negative/neutral):",
    keyword_name="sentiment"
)

# Apply multiple metataggers
chunks = topic_tagger(chunks)
chunks = sentiment_tagger(chunks)
```

### Domain-Specific Tagging
```python
# Medical text keyword extraction
medical_metatagger = KeywordMetatagger(
    client=client,
    system_prompt="""
    You are a medical text analysis expert. Extract keywords relevant to:
    - Medical conditions and symptoms
    - Treatments and procedures
    - Anatomical terms
    - Medical specialties
    """,
    user_prompt="Extract medical keywords and terms from this clinical text:",
    keyword_name="medical_keywords"
)

# Legal document tagging
legal_metatagger = KeywordMetatagger(
    client=client,
    system_prompt="Extract legal terms, case references, and key concepts.",
    user_prompt="Identify important legal keywords and references:",
    keyword_name="legal_terms"
)
```
