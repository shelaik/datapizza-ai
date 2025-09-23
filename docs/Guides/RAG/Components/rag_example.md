
# Real RAG Example

## Ingestion

```python
from datapizza.clients.openai_client import OpenAIClient
from datapizza.modules.captioners.llm_captioner import LLMCaptioner
from datapizza.modules.metatagger.keyword_metatagger import KeywordMetatagger
from datapizza.modules.parsers.text_parser import parse_text
from datapizza.modules.rewriters.tool_rewriter import ToolRewriter
from datapizza.modules.splitters.recursive_splitter import RecursiveSplitter
from datapizza.vectorstores import QdrantVectorstore
from datapizza.embedders import NodeEmbedder
from datapizza.vectorstores.vectorstore import VectorConfig

api_key="YOUR_API_KEY"

client = OpenAIClient(api_key=api_key, model="gpt-4o-mini")



# Process plain text
text_content = """
This is the first paragraph.
It contains multiple sentences.

This is the second paragraph.
It also has content.
"""


################ PARSING ################

document = parse_text(text_content, metadata={"source": "user_input"})


################ CAPTIONING ################

captioner = LLMCaptioner(
    client=client,
    max_workers=3,
    system_prompt_figure="Describe this image concisely",
    system_prompt_table="Describe this table's content and structure"
)
captioned_document = captioner(document)


################ SPLITTING ################

splitter = RecursiveSplitter(max_char=10, overlap=10)
chunks = splitter.run(captioned_document)



################ METATAGGING ################
tagger = KeywordMetatagger(
    client=client,
    max_workers=3,
    system_prompt="Extract 5 relevant keywords from the text",
    keyword_name="keywords"
)
tagged_chunks = tagger(chunks)



################ EMBEDDING ################

embedder = NodeEmbedder(
    client=client,
    model_name="text-embedding-3-small",
)
embedded_chunks = embedder(chunks)


################ VECTORSTORE ################

vectorstore = QdrantVectorstore(location=":memory:")
vectorstore.create_collection(collection_name="knowledge_base", vector_config=[VectorConfig(dimensions=1536, name="vector_name")])
vectorstore.add(embedded_chunks, collection_name="knowledge_base")



```

## Retrieval

```python




################ REWRITING ################

rewriter = ToolRewriter(
    client=client,
    system_prompt="rewrite the query to perform a better search in a vector database",
)

enhanced_query = rewriter.run("num paragraphs")


################ RETRIEVING ################

embedder = OpenAIClient(api_key=api_key, model="text-embedding-3-small")
query_embedding = embedder.embed(enhanced_query)
results = vectorstore.search(
    query_vector=query_embedding,
    collection_name="knowledge_base",
    k=2
)


################ RERANKING ################

reranker = TogetherReranker(
    api_key="your-key",
    model="BAAI/bge-reranker-large",
    top_n=10
)

final_results = reranker.run(enhanced_query, results)

print(f"Found {len(final_results)} similar documents")
print(final_results)

```
