import logging

import fastembed
from datapizza.core.embedder import BaseEmbedder

log = logging.getLogger(__name__)


class FastEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str,
        embedding_name: str | None = None,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        if embedding_name:
            self.embedding_name = embedding_name
        else:
            self.embedding_name = model_name

        self.cache_dir = cache_dir
        self.embedder = fastembed.SparseTextEmbedding(
            model_name=model_name, cache_dir=cache_dir
        )

    def embed(self, text: str) -> dict:
        embeddings = self.embedder.embed(text)
        return {self.embedding_name: embeddings}

    def a_embed(self, text: str) -> dict:
        embeddings = self.embedder.embed(text)
        return {self.embedding_name: embeddings}
