from datapizza.embedders.cohere import CohereEmbedder


def test_init_cohere_embedder():
    embedder = CohereEmbedder(api_key="test", base_url="test")
    assert embedder is not None
