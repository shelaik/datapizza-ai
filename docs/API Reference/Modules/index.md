# Modules

This section contains API reference documentation for all DataPizza AI modules. Modules are organized by functionality and include both core modules (included by default) and optional modules that require separate installation.

## Core Modules (Included by Default)

These modules are included with `datapizza-ai-core` and are available without additional installation:

- [Parsers](Parsers/) - Convert documents into structured Node representations
  - [TextParser](Parsers/text_parser.md) - Simple text parser for plain text
  - [AzureParser](Parsers/azure_parser.md) - Azure AI Document Intelligence parser (requires separate installation)
- [Captioners](captioners.md) - Generate captions and descriptions for content
- [Metatagger](metatagger.md) - Add metadata tags to content
- [Prompt](prompt.md) - Manage prompts and prompt templates
- [Rewriters](rewriters.md) - Transform and rewrite content
- [Splitters](Splitters/) - Split content into smaller chunks
  - [RecursiveSplitter](Splitters/recursive_splitter.md) - Recursive text splitting with multiple strategies
  - [TextSplitter](Splitters/text_splitter.md) - Basic text splitter
  - [NodeSplitter](Splitters/node_splitter.md) - Splitter for Node objects
  - [PDFImageSplitter](Splitters/pdf_image_splitter.md) - Specialized PDF content splitter
  - [BBoxMerger](Splitters/bbox_merger.md) - Bounding box merger utility
- [Treebuilder](treebuilder.md) - Build hierarchical tree structures from content

## Optional Modules (Separate Installation Required)

These modules require separate installation via pip:

- [Rerankers](Rerankers/) - Rerank and score content relevance
  - [CohereReranker](Rerankers/cohere_reranker.md) - Cohere API reranker
  - [TogetherReranker](Rerankers/together_reranker.md) - Together AI reranker

Each module page includes installation instructions and usage examples.