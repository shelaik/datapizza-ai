import uuid

import pytest
from datapizza.core.vectorstore import VectorConfig
from datapizza.type.type import Chunk, DenseEmbedding

from datapizza.vectorstores.qdrant import QdrantVectorstore


@pytest.fixture
def vectorstore() -> QdrantVectorstore:
    vectorstore = QdrantVectorstore(location=":memory:")
    vectorstore.create_collection(
        collection_name="test",
        vector_config=[VectorConfig(dimensions=1536, name="test")],
    )

    return vectorstore


def test_qdrant_vectorstore_init():
    vectorstore = QdrantVectorstore(location=":memory:")
    assert vectorstore is not None


def test_qdrant_vectorstore_add(vectorstore):
    chunks = [
        Chunk(
            id=str(uuid.uuid4()),
            text="Hello world",
            embeddings=[DenseEmbedding(name="test", vector=[0.0] * 1536)],
        )
    ]
    vectorstore.add(chunks, collection_name="test")

    assert (
        len(vectorstore.search(collection_name="test", query_vector=[0.0] * 1536)) == 1
    )


def test_qdrant_vectorstore_create_collection(vectorstore):
    vectorstore.create_collection(
        collection_name="test2",
        vector_config=[VectorConfig(dimensions=1536, name="test2")],
    )

    colls = vectorstore.get_collections()

    assert len(colls.collections) == 2


def test_delete_collection(vectorstore):
    vectorstore.create_collection(
        collection_name="deleteme",
        vector_config=[VectorConfig(dimensions=1536, name="test2")],
    )

    colls = vectorstore.get_collections()
    assert len(colls.collections) == 2
    vectorstore.delete_collection(collection_name="deleteme")

    colls = vectorstore.get_collections()
    assert len(colls.collections) == 1
