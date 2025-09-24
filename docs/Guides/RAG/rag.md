# Building a RAG System with DataPizza AI

This guide demonstrates how to build a complete RAG (Retrieval-Augmented Generation) system using DataPizza AI's pipeline architecture. We'll cover both the **ingestion pipeline** for processing and storing documents, and the **DagPipeline** for retrieval and response generation.

## Overview

A RAG system consists of two main phases:

1. **Ingestion**: Process documents, split them into chunks, generate embeddings, and store in a vector database
2. **Retrieval**: Query the vector database, retrieve relevant chunks, and generate responses

DataPizza AI provides specialized pipeline components for each phase:

- **IngestionPipeline**: Sequential processing for document ingestion
- **DagPipeline**: Graph-based processing for complex retrieval workflows

## Part 1: Document Ingestion Pipeline

The ingestion pipeline processes raw documents and stores them in a vector database. Here's a complete example:

### Basic Ingestion Setup

```python
from datapizza.pipeline import IngestionPipeline
from datapizza.modules.parsers import TextParser
from datapizza.modules.splitters import TextSplitter
from datapizza.embedders import ClientEmbedder
from datapizza.clients import ClientFactory
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.core.vectorstore import VectorConfig

# Initialize vector store
vectorstore = QdrantVectorstore(host="localhost", port=6333)
vectorstore.create_collection(
    "my_documents",
    vector_config=[VectorConfig(name="embedding", dimensions=1536)]
)

# Create embedding client
client_factory = ClientFactory()
embedder_client = client_factory.create_client(
    provider="openai",
    model="text-embedding-3-small",
    api_key="your-openai-api-key"
)

# Build ingestion pipeline
ingestion_pipeline = IngestionPipeline(
    modules=[
        TextParser(),                    # Parse documents
        TextSplitter(max_char=1000),    # Split into chunks
        ClientEmbedder(client=embedder_client)  # Generate embeddings
    ],
    vector_store=vectorstore,
    collection_name="my_documents"
)

# Ingest documents
documents = ["path/to/doc1.txt", "path/to/doc2.txt"]
ingestion_pipeline.run(documents, metadata={"source": "user_upload"})
```

### Advanced Ingestion with Multiple Components

For more complex document processing, you can add additional components:

```python
from datapizza.modules.captioners import LLMCaptioner
from datapizza.modules.splitters import NodeSplitter

# Create LLM client for captioning
llm_client = client_factory.create_client(
    provider="openai",
    model="gpt-4o-mini",
    api_key="your-openai-api-key"
)

# Advanced pipeline with captioning and node splitting
advanced_pipeline = IngestionPipeline(
    modules=[
        TextParser(),
        LLMCaptioner(
            client=llm_client,
            system_prompt_table="Generate concise captions for tables.",
            system_prompt_figure="Generate descriptive captions for figures."
        ),
        NodeSplitter(max_char=1500),
        ClientEmbedder(client=embedder_client)
    ],
    vector_store=vectorstore,
    collection_name="my_documents"
)

# Process documents with metadata
advanced_pipeline.run(
    "complex_document.pdf",
    metadata={
        "document_type": "research_paper",
        "author": "John Doe",
        "date": "2024-01-15"
    }
)
```

### Configuration-Based Ingestion

You can also define your pipeline using YAML configuration:

```yaml
# ingestion_config.yaml
constants:
  EMBEDDING_MODEL: "text-embedding-3-small"
  CHUNK_SIZE: 1000

ingestion_pipeline:
  clients:
    openai_embedder:
      provider: openai
      model: "${EMBEDDING_MODEL}"
      api_key: "your-openai-api-key"
    
  modules:
    - name: parser
      type: TextParser
      module: datapizza.modules.parsers
    - name: splitter
      type: TextSplitter
      module: datapizza.modules.splitters
      params:
        max_char: ${CHUNK_SIZE}
    - name: embedder
      type: ClientEmbedder
      module: datapizza.embedders
      params:
        client: openai_embedder

  vector_store:
    type: QdrantVectorstore
    module: datapizza.vectorstores.qdrant
    params:
      host: "localhost"
      port: 6333

  collection_name: "my_documents"
```

Load and use the configuration:

```python
pipeline = IngestionPipeline().from_yaml("ingestion_config.yaml")
pipeline.run(["document1.txt", "document2.txt"])
```

## Part 2: Retrieval with DagPipeline

The DagPipeline enables complex retrieval workflows with query rewriting, embedding, and response generation:

### Basic Retrieval Setup

```python
from datapizza.pipeline import DagPipeline
from datapizza.modules.rewriters import ToolRewriter
from datapizza.embedders import ClientEmbedder
from datapizza.modules.prompt import ChatPromptTemplate

# Create clients
rewriter_client = client_factory.create_client(
    provider="openai",
    model="gpt-4o-mini",
    api_key="your-openai-api-key"
)

response_client = client_factory.create_client(
    provider="openai",
    model="gpt-4o",
    api_key="your-openai-api-key"
)

# Create pipeline components
query_rewriter = ToolRewriter(
    client=rewriter_client,
    system_prompt="Rewrite user queries to improve retrieval accuracy."
)

embedder = ClientEmbedder(client=embedder_client)
retriever = vectorstore.as_retriever(collection_name="my_documents", k=5)

prompt_template = ChatPromptTemplate(
    system_prompt="You are a helpful assistant. Use the provided context to answer questions.",
    user_prompt="Context: {context}\n\nQuestion: {question}\n\nAnswer:"
)

# Build DAG pipeline
dag_pipeline = DagPipeline()

# Add modules
dag_pipeline.add_module("rewriter", query_rewriter)
dag_pipeline.add_module("embedder", embedder)
dag_pipeline.add_module("retriever", retriever)
dag_pipeline.add_module("prompt", prompt_template)
dag_pipeline.add_module("generator", response_client)

# Define connections
dag_pipeline.connect("rewriter", "embedder", target_key="input_data")
dag_pipeline.connect("embedder", "retriever", target_key="query_vector")
dag_pipeline.connect("retriever", "prompt", target_key="context")
dag_pipeline.connect("prompt", "generator", target_key="messages")

# Execute retrieval
query = "What are the main findings of the research?"
result = dag_pipeline.run({
    "rewriter": {"user_prompt": query},
    "prompt": {"question": query},
    "retriever": {"collection_name": "my_documents", "k": 3}
})

print(f"Generated response: {result['generator']}")
```
