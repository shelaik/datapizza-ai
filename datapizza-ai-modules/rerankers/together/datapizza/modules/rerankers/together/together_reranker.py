from datapizza.core.modules.reranker import Reranker
from datapizza.type import Chunk


class TogetherReranker(Reranker):
    """A reranker that uses the Together API to rerank documents."""

    def __init__(
        self,
        api_key: str,
        model: str,
        top_n: int = 10,
        threshold: float | None = None,
    ):
        """Initialize the TogetherReranker.

        Args:
            api_key (str): Together API key
            model (str): Model name to use for reranking
            top_n (int): Number of top documents to return
            threshold (Optional[float]): Minimum relevance score threshold. If None, no filtering is applied.
        """
        try:
            from together import Together
        except Exception as e:
            raise ValueError(f"Error importing together: {e}") from e

        self.client = Together(api_key=api_key)
        self.model = model
        self.top_n = top_n
        self.threshold = threshold

    def rerank(self, query: str, documents: list[Chunk]) -> list[Chunk]:
        """
        Rerank documents based on query.

        Args:
            query: The query to rerank documents by.
            documents: The documents to rerank.

        Returns:
            The reranked documents.
        """
        if not documents:
            return []

        top_n = min(self.top_n, len(documents))

        response = self.client.rerank.create(
            model=self.model,
            query=query,
            documents=[doc.text for doc in documents],
            top_n=top_n,
        )

        if response.results is None:
            return []

        # Create a list of (index, score) tuples from results
        scored_indices = [(r.index, r.relevance_score) for r in response.results]

        if self.threshold is not None:
            # Filter by threshold and sort by score
            scored_indices = [
                (idx, score) for idx, score in scored_indices if score >= self.threshold
            ]

        # Sort by score descending and extract just the indices
        sorted_indices = [
            idx for idx, _ in sorted(scored_indices, key=lambda x: x[1], reverse=True)
        ]

        # Return reranked documents in order
        return [documents[idx] for idx in sorted_indices]
