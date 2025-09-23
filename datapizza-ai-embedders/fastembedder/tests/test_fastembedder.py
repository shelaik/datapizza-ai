from datapizza.embedders.fastembedder import FastEmbedder


def test_init_fastembedder():
    embedder = FastEmbedder(model_name="Qdrant/bm25")
    assert embedder is not None
