import logging
import inspect
from typing import Any, Optional
import os
from qdrant_client import AsyncQdrantClient, QdrantClient, models

from datapizza.core.vectorstore import VectorConfig, Vectorstore
from datapizza.type import (
    Chunk,
    DenseEmbedding,
    Embedding,
    EmbeddingFormat,
    SparseEmbedding,
)

log = logging.getLogger(__name__)


class QdrantVectorstore(Vectorstore):
    """
    datapizza-ai implementation of a Qdrant vectorstore.

    Patch highlights:
    - Supports **embedded/local** mode (file-backed) *or* **server/remote** mode.
    - Accepts `local_path` (new), `location`, or `path` for embedded mode.
    - Detects installed `qdrant-client` constructor signature and adapts to
      either `location=` (newer) or `path=` (older) automatically.
    - Logs a clear message about detected version and chosen mode/argument.
    - Avoids passing host/port when using embedded mode (prevents IDNA errors).
    """

    def __init__(
        self,
        host: Optional[str] | None = None,
        port: int = 6333,
        api_key: Optional[str] | None = None,
        **kwargs,
    ):
        """
        Initialize the QdrantVectorstore.

        Args:
            host (str, optional): Remote Qdrant host. If omitted, embedded mode is used when a local path is provided.
            port (int, optional): Remote Qdrant port. Defaults to 6333.
            api_key (str, optional): API key (used in remote mode if auth enabled).
            **kwargs: May include one of `local_path`, `location`, or `path` for embedded mode,
                      plus any other Qdrant client kwargs.
        """
        # Embedded mode if any of these are provided
        has_embedded = any(k in kwargs for k in ("local_path", "location", "path"))
        if host is None and not has_embedded:
            raise ValueError(
                "Either host (remote) or local_path/location/path (embedded) must be provided"
            )

        self.client: QdrantClient
        self.a_client: Optional[AsyncQdrantClient]
        self.batch_size: int = 100
        self.host: Optional[str] | None = host
        self.port: int = port
        self.api_key: Optional[str] | None = api_key
        self.kwargs: dict[str, Any] = kwargs

        # Runtime flags
        self.is_embedded: bool = has_embedded and host is None
        self._embedded_param_used: Optional[str] = None  # 'location' or 'path'

    # ---------- Helpers ----------

    @staticmethod
    def _supports_param(cls, param_name: str) -> bool:
        try:
            sig = inspect.signature(cls.__init__)
            return param_name in sig.parameters
        except Exception:
            return False

    @staticmethod
    def _get_qdrant_client_version() -> str:
        try:
            from importlib.metadata import version, PackageNotFoundError
        except Exception:
            try:
                from importlib_metadata import version, PackageNotFoundError  # type: ignore
            except Exception:
                version = None  # type: ignore
                PackageNotFoundError = Exception  # type: ignore
        if version is None:
            return "unknown"
        try:
            return version("qdrant-client")
        except PackageNotFoundError:
            return "unknown"

    def _normalize_embedded_kwargs(self, for_async: bool = False) -> dict[str, Any]:
        """Return kwargs tailored for embedded mode, adapting to `location` vs `path`.
        Does *not* include host/port.
        """
        kwargs = dict(self.kwargs)

        # Prefer explicit local_path if provided
        if "local_path" in kwargs:
            local = kwargs.pop("local_path")
            # Choose the parameter name supported by the relevant client class
            ClientClass = AsyncQdrantClient if for_async else QdrantClient
            if self._supports_param(ClientClass, "location"):
                kwargs["location"] = local
                self._embedded_param_used = "location"
            elif self._supports_param(ClientClass, "path"):
                kwargs["path"] = local
                self._embedded_param_used = "path"
            else:
                raise ValueError(
                    "Installed qdrant-client does not accept 'location' or 'path' for embedded mode"
                )
            return kwargs

        # If caller supplied location or path already, make sure it's supported; if not, try to swap
        ClientClass = AsyncQdrantClient if for_async else QdrantClient
        if "location" in kwargs or "path" in kwargs:
            if "location" in kwargs and not self._supports_param(ClientClass, "location"):
                # try swap to path
                value = kwargs.pop("location")
                if self._supports_param(ClientClass, "path"):
                    kwargs["path"] = value
                    self._embedded_param_used = "path"
                else:
                    raise ValueError(
                        "'location' not supported by installed qdrant-client, and 'path' also unavailable"
                    )
            elif "path" in kwargs and not self._supports_param(ClientClass, "path"):
                # try swap to location
                value = kwargs.pop("path")
                if self._supports_param(ClientClass, "location"):
                    kwargs["location"] = value
                    self._embedded_param_used = "location"
                else:
                    raise ValueError(
                        "'path' not supported by installed qdrant-client, and 'location' also unavailable"
                    )
            else:
                self._embedded_param_used = "location" if "location" in kwargs else "path"
            return kwargs

        # Should not reach here if is_embedded True
        return kwargs

    # ---------- Client init ----------

    def get_client(self) -> QdrantClient:
        if not hasattr(self, "client"):
            self._init_client()
        return self.client

    def _get_a_client(self) -> AsyncQdrantClient:
        if not hasattr(self, "a_client") or self.a_client is None:
            self._init_a_client()
        assert self.a_client is not None, (
            "Async client not available in embedded mode on this environment. "
            "Use sync methods or run Qdrant server for async."
        )
        return self.a_client

    def _init_client(self):
        version_str = self._get_qdrant_client_version()
        if self.is_embedded:
            # --- NUOVO: ripulisce env che forzano il remote ---
            for k in ("QDRANT_URL", "QDRANT__URL", "QDRANT_HOST", "QDRANT__HOST"):
                if os.getenv(k):
                    log.warning("Embedded mode: clearing env %s='%s'", k, os.getenv(k))
                    os.environ.pop(k, None)
    
            kwargs = self._normalize_embedded_kwargs(for_async=False)
            # Non passare host/port/api_key in embedded
            kwargs.pop("host", None)
            kwargs.pop("port", None)
            kwargs.pop("api_key", None)
    
            self.client = QdrantClient(**kwargs)
            log.info(
                "Qdrant client %s detected; initializing in EMBEDDED mode using %s=...",
                version_str,
                self._embedded_param_used or "location/path",
            )
    
            # --- NUOVO: verifica che sia davvero Local ---
            backend_name = getattr(getattr(self.client, "_client", None), "__class__", type(None)).__name__
            if backend_name.lower().find("local") < 0:
                raise RuntimeError(
                    f"Expected embedded backend but got '{backend_name}'. "
                    f"Kwargs={kwargs}. Check env and qdrant-client version."
                )
    
        else:
            self.client = QdrantClient(
                host=self.host, port=self.port, api_key=self.api_key, **self.kwargs
            )
            log.info(
                "Qdrant client %s detected; initializing in REMOTE mode (host=%s, port=%s)",
                version_str,
                self.host,
                self.port,
            )


    def _init_a_client(self):
        version_str = self._get_qdrant_client_version()
        if self.is_embedded:
            # Try to init async embedded; if it fails, keep None and warn
            try:
                kwargs = self._normalize_embedded_kwargs(for_async=True)
                kwargs.pop("host", None)
                kwargs.pop("port", None)
                kwargs.pop("api_key", None)
                self.a_client = AsyncQdrantClient(**kwargs)
                log.info(
                    "Qdrant client %s detected; async EMBEDDED client initialized using %s=...",
                    version_str,
                    self._embedded_param_used or "location/path",
                )
            except Exception as e:
                self.a_client = None
                log.warning(
                    "Async embedded client not available (qdrant-client %s). "
                    "Falling back to sync-only. Reason: %s",
                    version_str,
                    e,
                )
        else:
            self.a_client = AsyncQdrantClient(
                host=self.host, port=self.port, api_key=self.api_key, **self.kwargs
            )
            log.info(
                "Qdrant client %s detected; initializing async REMOTE client (host=%s, port=%s)",
                version_str,
                self.host,
                self.port,
            )

    # ---------- CRUD ----------

    def add(self, chunk: Chunk | list[Chunk], collection_name: str | None = None):
        """Add a single chunk or list of chunks to the vectorstore.
        Args:
            chunk (Chunk | list[Chunk]): The chunk or list of chunks to add.
            collection_name (str, optional): The name of the collection to add the chunks to. Defaults to None.
        """
        client = self.get_client()

        if not collection_name:
            raise ValueError("Collection name must be set")

        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        points = []

        for ch in chunks:
            points.append(self._process_chunk(ch))

        # TODO: Process in batches
        for p in points:
            try:
                client.upsert(collection_name=collection_name, points=[p], wait=True)
            except Exception as e:
                log.error(f"Failed to add points to Qdrant: {e!s}")
                raise e

    async def a_add(
        self, chunk: Chunk | list[Chunk], collection_name: str | None = None
    ):
        client = self._get_a_client()

        if not collection_name:
            raise ValueError("Collection name must be set")

        chunks = [chunk] if isinstance(chunk, Chunk) else chunk
        points = []

        for ch in chunks:
            points.append(self._process_chunk(ch))

        # TODO: Process in batches
        for p in points:
            try:
                await client.upsert(
                    collection_name=collection_name, points=[p], wait=True
                )
            except Exception as e:
                log.error(f"Failed to add points to Qdrant: {e!s}")
                raise e

    def _process_chunk(self, chunk: Chunk) -> models.PointStruct:
        """Process a chunk into a Qdrant point."""
        if not chunk.embeddings:
            raise ValueError("Chunk must have an embedding")

        vector: Any = {}
        if len(chunk.embeddings) == 1:
            if isinstance(chunk.embeddings[0], DenseEmbedding):
                vector = chunk.embeddings[0].vector
            elif isinstance(chunk.embeddings[0], SparseEmbedding):
                vector = models.SparseVector(
                    values=chunk.embeddings[0].values,
                    indices=chunk.embeddings[0].indices,
                )
            else:
                raise ValueError(
                    f"Unsupported embedding type: {type(chunk.embeddings[0])}"
                )

        else:
            for v in chunk.embeddings:
                if isinstance(v, DenseEmbedding):
                    vector[v.name] = v.vector
                elif isinstance(v, SparseEmbedding):
                    vector[v.name] = models.SparseVector(
                        values=v.values, indices=v.indices
                    )
                else:
                    raise ValueError(f"Unsupported embedding type: {type(v)}")

        return models.PointStruct(
            id=str(chunk.id),
            payload={
                "text": chunk.text,
                **chunk.metadata,
            },
            vector=vector,  # type: ignore
        )

    def update(self, collection_name: str, payload: dict, points: list[int] | list[str], **kwargs):
        client = self.get_client()
        client.overwrite_payload(
            collection_name=collection_name,
            payload=payload,
            points=points,  # type: ignore
            **kwargs,
        )

    def retrieve(self, collection_name: str, ids: list[str], **kwargs) -> list[Chunk]:
        """Retrieve chunks from a collection by their IDs.
        Args:
            collection_name (str): The name of the collection to retrieve the chunks from.
            ids (list[str]): The IDs of the chunks to retrieve.
            **kwargs: Additional keyword arguments to pass to the Qdrant client.
        Returns:
            list[Chunk]: The list of chunks retrieved from the collection.
        """
        client = self.get_client()
        return self._point_to_chunk(
            client.retrieve(
                collection_name=collection_name,
                ids=ids,
                **kwargs,
            )
        )

    def remove(self, collection_name: str, ids: list[str], **kwargs):
        """Remove chunks from a collection by their IDs.
        Args:
            collection_name (str): The name of the collection to remove the chunks from.
            ids (list[str]): The IDs of the chunks to remove.
            **kwargs: Additional keyword arguments to pass to the Qdrant client.
        """
        client = self.get_client()
        client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(
                points=ids,  # type: ignore
            ),
            **kwargs,
        )

    def search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        """
        Search for chunks in a collection by their query vector.

        Args:
            collection_name (str): The name of the collection to search in.
            query_vector (list[float]): The query vector to search for.
            k (int, optional): The number of results to return. Defaults to 10.
            vector_name (str, optional): The name of the vector to search for. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the Qdrant client.

        Returns:
            list[Chunk]: The list of chunks found in the collection.
        """
        client = self.get_client()

        qry = (vector_name, query_vector) if vector_name else query_vector

        hits = client.search(
            collection_name=collection_name,
            query_vector=qry,
            limit=k,  # Return k closest points
            **kwargs,
        )
        return self._point_to_chunk(hits)

    async def a_search(
        self,
        collection_name: str,
        query_vector: list[float],
        k: int = 10,
        vector_name: str | None = None,
        **kwargs,
    ) -> list[Chunk]:
        """Search for chunks in a collection by their query vector."""
        client = self._get_a_client()

        qry = (vector_name, query_vector) if vector_name else query_vector

        hits = await client.search(
            collection_name=collection_name,
            query_vector=qry,
            limit=k,  # Return k closest points
            **kwargs,
        )
        return self._point_to_chunk(hits)

    def get_collections(self):
        """Get all collections in Qdrant."""
        client = self.get_client()
        return client.get_collections()

    def create_collection(
        self, collection_name: str, vector_config: list[VectorConfig], **kwargs
    ):
        """Create a new collection in Qdrant if it doesn't exist with the specified vector configurations

        Args:
            collection_name: Name of the collection to create
            vector_config: List of vector configurations specifying dimensions and distance metrics
            **kwargs: Additional arguments to pass to Qdrant's create_collection
        """

        client = self.get_client()

        if client.collection_exists(collection_name):
            log.warning(
                f"Collection {collection_name} already exists, skipping creation"
            )
            return

        sparse_config: (
            dict[str, models.SparseVectorParams] | models.SparseVectorParams | None
        ) = None
        config = None
        try:
            if len(vector_config) == 1:
                if vector_config[0].format == EmbeddingFormat.DENSE:
                    config = models.VectorParams(
                        size=vector_config[0].dimensions,
                        distance=vector_config[0].distance.value,  # type: ignore
                    )
                elif vector_config[0].format == EmbeddingFormat.SPARSE:
                    sparse_config = models.SparseVectorParams()

            else:
                # Multiple vector configurations
                config = {
                    v.name: models.VectorParams(
                        size=v.dimensions,
                        distance=v.distance.value,  # type: ignore
                    )
                    for v in vector_config
                    if v.format == EmbeddingFormat.DENSE
                }
                sparse_config = {
                    v.name: models.SparseVectorParams()
                    for v in vector_config
                    if v.format == EmbeddingFormat.SPARSE
                }

            client.create_collection(
                collection_name=collection_name,
                vectors_config=config,
                sparse_vectors_config=sparse_config,  # type: ignore
                **kwargs,
            )
        except Exception as e:
            log.error(f"Failed to create collection {collection_name}: {e!s}")
            raise e

    def delete_collection(self, collection_name: str, **kwargs):
        """Delete a collection in Qdrant."""
        client = self.get_client()
        client.delete_collection(collection_name=collection_name, **kwargs)

    def dump_collection(
        self,
        collection_name: str,
        page_size: int = 100,
        with_vectors: bool = False,
    ) -> "Generator[Chunk, None, None]":
        """
        Dumps all points from a collection in a chunk-wise manner.

        Args:
            collection_name: Name of the collection to dump.
            page_size: Number of points to retrieve per batch.
            with_vectors: Whether to include vectors in the dumped chunks.

        Yields:
            Chunk: A chunk object from the collection.
        """
        client = self.get_client()
        next_page_offset = None

        while True:
            points, next_page_offset = client.scroll(
                collection_name=collection_name,
                limit=page_size,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=with_vectors,
            )

            if not points:
                break

            yield from self._point_to_chunk(points)

            if next_page_offset is None:
                break

    def _point_to_chunk(self, points) -> list[Chunk]:
        """
        Convert Qdrant points to Chunk objects.

        Args:
            points: List of Qdrant point objects

        Returns:
            List of Chunk objects with appropriate embeddings
        """
        chunks = []

        for point in points:
            vector = point.vector
            embeddings: list[Embedding] = []

            # Handle dictionary of named vectors
            if isinstance(vector, dict):
                for name, vec in vector.items():
                    if isinstance(vec, models.SparseVector):
                        embeddings.append(
                            SparseEmbedding(
                                name=name,
                                values=vec.values,
                                indices=vec.indices,
                            )
                        )
                    elif isinstance(vec, list):
                        embeddings.append(DenseEmbedding(name=name, vector=vec))
            # Handle single dense vector (list)
            elif isinstance(vector, list):
                embeddings.append(DenseEmbedding(name="dense", vector=vector))
            # Handle single sparse vector
            elif isinstance(vector, models.SparseVector):
                embeddings.append(
                    SparseEmbedding(
                        name="sparse", values=vector.values, indices=vector.indices
                    )
                )
            elif vector is None:
                embeddings = []
            else:
                raise ValueError(f"Unsupported vector type: {type(vector)}")

            chunks.append(
                Chunk(
                    id=point.id,
                    metadata=point.payload,
                    text=point.payload["text"],
                    embeddings=embeddings,
                )
            )

        return chunks
