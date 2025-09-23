# Captioners

Captioners are pipeline components that generate descriptive captions for media elements (images, tables, figures) using language models. They enhance MediaNodes with AI-generated descriptions for better searchability and understanding.

## Available Captioners

### LLMCaptioner

Generates captions for images and tables using any LLM client with vision capabilities.

```python
from datapizza.modules.captioners import LLMCaptioner
from datapizza.clients.openai_client import OpenAIClient

client = OpenAIClient(api_key="your-key", model="gpt-4o")
captioner = LLMCaptioner(
    client=client,
    max_workers=3,
    system_prompt_figure="Describe this image concisely",
    system_prompt_table="Describe this table's content and structure"
)

captioned_document = captioner(document_node)
```

**Parameters:**

- `client` (Client): Vision-capable LLM client for caption generation
- `max_workers` (int): Concurrent processing threads (default: 3)
- `system_prompt_figure` (str, optional): Custom prompt for image captioning
- `system_prompt_table` (str, optional): Custom prompt for table captioning

**Features:**

- Concurrent processing of multiple media elements
- Wraps captions in XML-style tags for structured output
- Preserves original media and metadata
- Supports both sync and async processing
- Works with any vision-capable LLM client

## Processing Behavior

The captioner:

1. Identifies all MediaNode objects in the document tree
2. Processes them concurrently using ThreadPoolExecutor
3. Generates captions based on node type (FIGURE or TABLE)
4. Wraps content with structured tags: `<figure>[caption]</figure>` or `<table>[caption]</table>`

## Async Processing
```python
async def process_media():
    captioned_doc = await captioner.a_run(document)
    return captioned_doc
```

## Document Processing Pipeline

```python
import os

from datapizza.clients.openai_client import OpenAIClient
from datapizza.modules.captioners import LLMCaptioner
from datapizza.type import Media, MediaNode, Node, NodeType
from dotenv import load_dotenv

load_dotenv()


document = Node(children=[
    Node(content="This is a very long text that should be split into chunks", node_type=NodeType.PARAGRAPH),
    MediaNode(media=Media(source="https://images.google.com/images/branding/googlelogo/2x/googlelogo_light_color_272x92dp.png", media_type="image", source_type="url"), node_type=NodeType.FIGURE),
])

client = OpenAIClient(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o")
captioner = LLMCaptioner(client=client)
enhanced_document = captioner(document)

# Media nodes now have descriptive captions
for child in enhanced_document.children:
    if child.node_type == NodeType.FIGURE:
        print(f"Caption: {child.content}")
    
# Caption:  <figure> [This is the Google logo.] </figure>
```
