# Ingestion Pipeline


The `IngestionPipeline` provides a streamlined way to process documents, transform them into nodes (chunks of text with metadata), generate embeddings, and optionally store them in a vector database. It allows chaining various components like parsers, captioners, splitters, and embedders to create a customizable document processing workflow.


## Core Concepts

-   **Components**: These are the processing steps in the pipeline, typically inheriting from `datapizza.core.models.PipelineComponent`. Each component implements a `process` method to perform a specific task like parsing a document, splitting text, or generating embeddings. Components are executed sequentially via their `__call__` method in the order they are provided.
-   **Vector Store**: An optional component responsible for storing the final nodes and their embeddings.
-   **Nodes**: The fundamental unit of data passed between components. A node usually represents a chunk of text (e.g., a paragraph, a table summary) along with its associated metadata and embeddings.

## Available Components

The pipeline typically supports components for:

1.  [**Parsers**](../../API%20Reference/Modules/Parsers/index.md): Convert raw documents (PDF, DOCX, etc.) into structured `Node` objects (e.g., `AzureParser`, `UnstructuredParser`).
2.  [**Captioners**](../../API%20Reference/Modules/captioners.md): Enhance nodes representing images or tables with textual descriptions using models like LLMs (e.g., `LLMCaptioner`).
3.  [**Splitters**](../../API%20Reference/Modules/Splitters/index.md): Divide nodes into smaller chunks based on their content (e.g., `NodeSplitter`, `PdfImageSplitter`).
4.  [**Embedders**](../../API%20Reference/Embedders/openai_embedder.md): Create chunk embeddings for semantic search and similarity matching (e.g., `NodeEmbedder`, `ClientEmbedder`).
     - [`ChunkEmbedder`](../../API%20Reference/Embedders/chunk_embedder.md): Batch processing for efficient embedding of multiple nodes.
5.  [**Vector Stores**](../../API%20Reference/Vectorstore/qdrant_vectorstore.md): Store and retrieve embeddings efficiently using vector databases (e.g., `QdrantVectorstore`).

Refer to the specific documentation for each component type (e.g., in `datapizza.parsers`, `datapizza.embedders`) for details on their specific parameters and usage. Remember that pipeline components typically inherit from `PipelineComponent` and implement the `_run` method.


## Configuration Methods

There are two main ways to configure and use the `IngestionPipeline`:

### 1. Programmatic Configuration

Define and configure the pipeline directly within your Python code. This offers maximum flexibility.

```python
import base64

from datapizza.clients import Client, OpenAIClient
from datapizza.core.models import PipelineComponent
from datapizza.embedders import ClientEmbedder, NodeEmbedder
from datapizza.modules.splitters import TextSplitter
from datapizza.pipeline.pipeline import IngestionPipeline
from datapizza.type import Media, MediaBlock, TextBlock
from datapizza.vectorstores.qdrant_vectorstore import QdrantVectorstore, VectorConfig
from dotenv import load_dotenv


client = OpenAIClient(
    api_key="YOUR_API_KEY",
    model="gpt-4o-mini",
)

embedder = NodeEmbedder(
    client=client, model_name="text-embedding-3-small", embedding_name="small"
)

class MyCustomParser(PipelineComponent):
    def __init__(self, client: Client):
        self.client = client

    def run(self, file_path: str):

        with open(file_path, "rb") as file:
            file_data = file.read()
            base64_data = base64.b64encode(file_data).decode("utf-8")

        media = Media(
            extension="pdf",
            media_type="pdf",
            source_type="base64",
            source=base64_data,
        )
        block = MediaBlock(
            media=media,
        )
        response = client.invoke([TextBlock(content="Estrai tutto il testo che vedi in questo documento"), block])
        return response.text


vector_store = QdrantVectorstore(
    location=":memory:"
)
vector_store.create_collection(collection_name="datapizza", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])

pipeline = IngestionPipeline(
    modules=[
        MyCustomParser(client=client),
        TextSplitter(max_char=2000, overlap=100),
        embedder, 
    ],
    vector_store=vector_store,
    collection_name="datapizza",
)

pipeline.run(file_path="path_to_your_file.pdf")

text_embedder = ClientEmbedder(client=client, model_name="text-embedding-3-small", embedding_name="small")
query = "Who won the world cup in 2006?"
results = vector_store.search(query_vector=text_embedder.embed(query), collection_name="datapizza", k=4)
print(results)

```


### 2. YAML Configuration

Define the entire pipeline structure, components, and their parameters in a YAML file. This is useful for managing configurations separately from code.

```python
from datapizza.pipeline.pipeline import IngestionPipeline
import os

# Load pipeline from YAML
pipeline = IngestionPipeline().from_yaml("path/to/your/config.yaml")

# Run the pipeline (Ensure necessary ENV VARS for the YAML config are set)
pipeline.run(file_path="path/to/your/document.pdf")
```

#### Example YAML Configuration (`config.yaml`)

```yaml
ingestion_pipeline:
  clients: # Define reusable clients (e.g., LLMs, Embedders)
    openai_chat_client:
      provider: openai
      model: "gpt-4o-mini"
      api_key: "${OPENAI_API_KEY}" # Use environment variables for secrets
    openai_embed_client:
      provider: openai
      model: "text-embedding-3-small"
      api_key: "${OPENAI_API_KEY}"

  modules: # Define the sequence of processing components
    - name: parser # Optional descriptive name
      type: AzureParser # Component class name
      module: datapizza.modules.parsers # Python module path where the class resides
      params: # Parameters passed to the component's __init__
        api_key: "${AZURE_DOC_INTEL_KEY}"
        endpoint: "${AZURE_DOC_INTEL_ENDPOINT}"
        result_type: "markdown" # Or "text"
    - name: captioner
      type: LLMCaptioner
      module: datapizza.modules.captioners
      params:
        client: openai_chat_client # Reference a defined client
        system_prompt_table: "Summarize the key information in this table concisely."
        system_prompt_figure: "Provide a brief description of this image."
    - name: splitter
      type: NodeSplitter # Using NodeSplitter as an example
      module: datapizza.modules.splitters
      params:
        max_char: 4000
        overlap: 200
    - name: embedder
      type: NodeEmbedder # Use NodeEmbedder for processing node lists
      module: datapizza.embedders
      params:
        client: openai_embed_client # Reference the embedding client
        embedding_name: "openai_small" # Name for the embedding vector in Node metadata

  vector_store: # Optional: Define the vector store for storing results
    type: QdrantVectorstore
    module: datapizza.vectorstores
    params:
      host: "${QDRANT_HOST}" # Use environment variables
      port: ${QDRANT_PORT} # Environment variables can be used directly if numeric
      # api_key: "${QDRANT_API_KEY}" # If authentication is needed
      # prefer_grpc: true

  collection_name: "my-yaml-collection" # Required if vector_store is defined
```

**Key points for YAML configuration:**

-   **Environment Variables**: Use `${VAR_NAME}` syntax within strings to securely load secrets or configuration from environment variables. Ensure these variables are set in your execution environment.
-   **Clients**: Define shared clients (like `OpenAIClient`) under the `clients` key and reference them by name within module `params`.
-   **Modules**: List components under `modules`. Each requires `type` (class name) and `module` (Python path to the class). `params` are passed to the component's constructor (`__init__`). Components should generally inherit from `PipelineComponent`.
-   **Vector Store**: Configure the optional vector store similarly to modules.
-   **Collection Name**: Must be provided if a `vector_store` is configured.


## Pipeline Execution (`run` method)
```python
pipeline.run(file_path=f, metadata={"name": f, "type": "md"})
```
### Async Execution (`a_run` method)

IngestionPipeline support async run
*NB:* Every modules should implement `_a_run` method to run the async pipeline.

```python
await pipeline.a_run(file_path=f, metadata={"name": f, "type": "md"})
```

