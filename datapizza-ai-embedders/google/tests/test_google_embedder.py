import pytest
from datapizza.embedders.google import GoogleEmbedder


def test_google_embedder_init():
    embedder = GoogleEmbedder(api_key="test-key")
    assert embedder.api_key == "test-key"
    assert embedder.client is None
    assert embedder.a_client is None