import base64
import copy
import logging
import mimetypes
import os

import cohere
from datapizza.core.models import PipelineComponent
from datapizza.type.type import Chunk, DenseEmbedding, Media, MediaBlock

log = logging.getLogger(__name__)


class CohereImageClient(PipelineComponent):
    """Client for interacting with Cohere API for image embeddings.

    Can be initialized with pre-existing Cohere clients (sync and/or async)
    or with API credentials for lazy client initialization.
    """

    DEFAULT_MODEL = "embed-v4.0"
    DEFAULT_INPUT_TYPE = "image_document"

    def __init__(
        self,
        cohere_client: cohere.Client
        | cohere.AsyncClient
        | None = None,  # Sync or async client (for backward compatibility)
        a_cohere_client: cohere.AsyncClient
        | None = None,  # Dedicated asynchronous client
        api_key: str | None = None,
        base_url: str | None = None,  # For Azure or custom deployments
        model: str = DEFAULT_MODEL,
        embedding_type: str = "float",
    ):
        """Initialize the Cohere Image Client.

        Args:
            cohere_client: Pre-existing Cohere client (sync or async). For backward compatibility,
                          if an AsyncClient is passed here, it will be used as the async client.
            a_cohere_client: Pre-existing async Cohere client. Takes precedence over async client
                           passed to cohere_client parameter.
            api_key: Cohere API key. If not provided, will try COHERE_API_KEY or AZURE_COHERE_API_KEY
                    environment variables.
            base_url: Base URL for custom/Azure deployments. If not provided, will try
                     AZURE_COHERE_ENDPOINT environment variable.
            model: Model name to use for embeddings.
            embedding_type: Type of embeddings to retrieve ('float', 'int8', etc.).
        """
        # Initialize clients
        self.client: cohere.Client | None = None
        self.a_client: cohere.AsyncClient | None = None

        # Handle the main cohere_client parameter (backward compatibility)
        if cohere_client:
            if isinstance(cohere_client, cohere.AsyncClient):
                # If an async client was passed to the main parameter, use it as async client
                self.a_client = cohere_client
            else:
                # Otherwise, it's a sync client
                self.client = cohere_client

        # Handle dedicated async client parameter
        if a_cohere_client:
            self.a_client = a_cohere_client

        # Store credentials for lazy client initialization
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.embedding_type = embedding_type

        # If no clients are provided and no API key, try to get from environment
        if not cohere_client and not a_cohere_client and not api_key:
            resolved_api_key = os.environ.get("COHERE_API_KEY") or os.environ.get(
                "AZURE_COHERE_API_KEY"
            )
            if not resolved_api_key:
                raise ValueError(
                    "Either cohere_client/a_cohere_client must be provided, or API key must be "
                    "provided via api_key parameter or environment variables "
                    "(COHERE_API_KEY or AZURE_COHERE_API_KEY)"
                )
            self.api_key = resolved_api_key

    def _create_data_uri(self, media: Media) -> str:
        """Creates a data URI from a Media object."""
        if media.media_type != "image":
            raise ValueError(f"Media type must be 'image', got '{media.media_type}'")

        if media.source_type == "base64":
            # Already in base64 format
            if media.extension:
                mime_type = f"image/{media.extension.lstrip('.')}"
            else:
                mime_type = "image/jpeg"  # Default fallback
            return f"data:{mime_type};base64,{media.source}"

        elif media.source_type == "path":
            # Handle file path
            image_path = str(media.source)
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type or not mime_type.startswith("image"):
                raise ValueError(
                    f"Could not determine image type or invalid image type for {image_path}"
                )
            try:
                with open(image_path, "rb") as f:
                    image_data = f.read()
            except FileNotFoundError as e:
                raise FileNotFoundError(f"Image file not found at {image_path}") from e
            except Exception as e:
                raise OSError(f"Error reading image file {image_path}: {e}") from e

            base64_encoded_data = base64.b64encode(image_data)
            base64_string = base64_encoded_data.decode("utf-8")
            return f"data:{mime_type};base64,{base64_string}"

        elif media.source_type == "url":
            # For URLs, we'd need to download the image first
            # For now, raise an error as this would require additional dependencies
            raise NotImplementedError(
                "URL source type not yet supported for Cohere embeddings"
            )

        else:
            raise ValueError(f"Unsupported source type: {media.source_type}")

    def _get_client(self) -> cohere.Client:
        """Get the synchronous client"""
        if not self.client:
            self._set_client()

        if not self.client:
            raise ValueError("Cohere client is not initialized properly.")

        return self.client

    def _set_client(self):
        """Set up the synchronous client"""
        if not self.client:
            resolved_api_key = (
                self.api_key
                or os.environ.get("COHERE_API_KEY")
                or os.environ.get("AZURE_COHERE_API_KEY")
            )
            if not resolved_api_key:
                raise ValueError(
                    "Cohere API key must be provided or set in environment variables "
                    "(COHERE_API_KEY or AZURE_COHERE_API_KEY)"
                )

            resolved_base_url = self.base_url or os.environ.get("AZURE_COHERE_ENDPOINT")
            client_args: dict = {"api_key": resolved_api_key}
            if resolved_base_url:
                client_args["base_url"] = resolved_base_url
            self.client = cohere.ClientV2(**client_args)

    def _get_a_client(self) -> cohere.AsyncClient:
        """Get the asynchronous client"""
        if not self.a_client:
            self._set_a_client()

        if not self.a_client:
            raise ValueError("Cohere async client is not initialized properly.")

        return self.a_client

    def _set_a_client(self):
        """Set up the asynchronous client"""
        if not self.a_client:
            resolved_api_key = (
                self.api_key
                or os.environ.get("COHERE_API_KEY")
                or os.environ.get("AZURE_COHERE_API_KEY")
            )
            if not resolved_api_key:
                raise ValueError(
                    "Cohere API key must be provided or set in environment variables "
                    "(COHERE_API_KEY or AZURE_COHERE_API_KEY)"
                )

            resolved_base_url = self.base_url or os.environ.get("AZURE_COHERE_ENDPOINT")
            client_args: dict = {"api_key": resolved_api_key}
            if resolved_base_url:
                client_args["base_url"] = resolved_base_url
            self.a_client = cohere.AsyncClientV2(**client_args)

    def embed(self, media: Media) -> list[float]:
        """
        Generates embeddings for a single image using the Cohere API.

        Args:
            media: A Media object containing the image data.

        Returns:
            A single embedding vector.

        Raises:
            ValueError: If the Media object is invalid or embedding extraction fails.
            AttributeError: If the requested embedding type is not found in the Cohere response.
            Exception: If the Cohere API call fails.
        """
        image_uri = self._create_data_uri(media)

        try:
            client = self._get_client()
            response = client.embed(
                texts=[],  # Required but not used for image embeddings
                images=[image_uri],  # Only one image per call
                model=self.model,
                input_type=self.DEFAULT_INPUT_TYPE,
                embedding_types=[self.embedding_type],
            )

            # Extract embeddings based on the requested type
            embeddings_data = response.embeddings

            if not hasattr(embeddings_data, self.embedding_type):
                raise AttributeError(
                    f"Cohere response does not contain image embeddings of type '{self.embedding_type}'."
                )

            vector = getattr(embeddings_data, self.embedding_type)[
                0
            ]  # Get the first (only) vector

            return vector

        except Exception as e:
            log.error(f"Error during Cohere embedding generation for media: {e}")
            raise e

    async def a_embed(self, media: Media) -> list[float]:
        """
        Asynchronously generates embeddings for a single image using the Cohere API.

        Args:
            media: A Media object containing the image data.

        Returns:
            A single embedding vector.

        Raises:
            ValueError: If the Media object is invalid or embedding extraction fails.
            AttributeError: If the requested embedding type is not found in the Cohere response.
            Exception: If the Cohere API call fails.
        """
        image_uri = self._create_data_uri(media)

        try:
            a_client = self._get_a_client()
            response = await a_client.embed(
                texts=[],  # Required but not used for image embeddings
                images=[image_uri],  # Only one image per call
                model=self.model,
                input_type=self.DEFAULT_INPUT_TYPE,
                embedding_types=[self.embedding_type],
            )

            # Extract embeddings based on the requested type
            embeddings_data = response.embeddings

            if not hasattr(embeddings_data, self.embedding_type):
                raise AttributeError(
                    f"Cohere response does not contain image embeddings of type '{self.embedding_type}'."
                )

            vector = getattr(embeddings_data, self.embedding_type)[
                0
            ]  # Get the first (only) vector

            return vector

        except Exception as e:
            log.error(f"Error during Cohere embedding generation for media: {e}")
            raise e

    def _run(self, media: Media, **kwargs) -> list[float] | list[int]:
        """
        Pipeline component sync run method that generates embeddings for a single image.

        Args:
            media: A Media object containing the image data.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A single embedding vector.
        """
        return self.embed(media)

    async def _a_run(self, media: Media, **kwargs) -> list[float] | list[int]:
        """
        Pipeline component async run method that generates embeddings for a single image.

        Args:
            media: A Media object containing the image data.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            A single embedding vector.
        """
        return await self.a_embed(media)


class NodeImageEmbedder(PipelineComponent):
    """Embeds image chunks using a CohereImageClient."""

    DEFAULT_EMBEDDING_NAME = "cohere_image_embedding"

    def __init__(
        self,
        client: CohereImageClient,
        embedding_name: str = DEFAULT_EMBEDDING_NAME,
    ):
        self.client = client
        self.embedding_name = embedding_name

    def _run(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Generates embeddings for a list of image Chunks.

        Args:
            chunks: A list of Chunk objects. Each chunk must have a `media_block` key
                    in its `metadata` dictionary containing a MediaBlock.

        Returns:
            The same list of Chunk objects, with the 'embeddings' list updated.
        """
        processed_chunks = copy.deepcopy(chunks)
        if not processed_chunks:
            return []

        # Process each chunk individually since we're now handling single images
        for chunk in processed_chunks:
            if not hasattr(chunk, "metadata") or not isinstance(chunk.metadata, dict):
                log.warning(
                    "Warning: Chunk is missing 'metadata' dictionary. Skipping."
                )
                continue

            media_block = chunk.metadata.get("media_block")
            if not media_block:
                log.warning(
                    f"Warning: Chunk with ID {chunk.id if hasattr(chunk, 'id') else 'N/A'} is missing 'media_block' in metadata. Skipping."
                )
                continue

            if not isinstance(media_block, MediaBlock):
                log.warning(
                    f"Warning: 'media_block' in chunk metadata is not a MediaBlock instance for chunk ID {chunk.id if hasattr(chunk, 'id') else 'N/A'}. Skipping."
                )
                continue

            # Validate that it's an image media block
            if media_block.media.media_type != "image":
                log.warning(
                    f"Warning: MediaBlock media_type is '{media_block.media.media_type}', expected 'image' for chunk ID {chunk.id if hasattr(chunk, 'id') else 'N/A'}. Skipping."
                )
                continue

            try:
                # Generate embedding for this single image
                embedding_vector = self.client.embed(media_block.media)

                new_embedding = DenseEmbedding(
                    name=self.embedding_name, vector=embedding_vector
                )
                if not hasattr(chunk, "embeddings") or chunk.embeddings is None:
                    chunk.embeddings = []
                    chunk.embeddings.append(new_embedding)

                # Remove the media_block from metadata after successful embedding generation
                # since it's no longer needed and can't be serialized to vectorstore
                chunk.metadata.pop("media_block", None)

            except (
                OSError,
                ValueError,
                FileNotFoundError,
                AttributeError,
                NotImplementedError,
            ) as e:
                # These errors are more likely to be raised by client.embed or _create_data_uri
                log.error(
                    f"Error processing image chunk {chunk.id if hasattr(chunk, 'id') else 'N/A'}: {e}. Skipping."
                )
                raise e
            except Exception as e:
                log.error(
                    f"Unexpected error during embedding generation for chunk {chunk.id if hasattr(chunk, 'id') else 'N/A'}: {e}. Skipping."
                )
                raise e

        return processed_chunks

    async def _a_run(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Asynchronously generates embeddings for a list of image Chunks.

        Args:
            chunks: A list of Chunk objects. Each chunk must have a `media_block` key
                    in its `metadata` dictionary containing a MediaBlock.

        Returns:
            The same list of Chunk objects, with the 'embeddings' list updated.
        """
        processed_chunks = copy.deepcopy(chunks)
        if not processed_chunks:
            return []

        # Process each chunk individually since we're now handling single images
        for chunk in processed_chunks:
            if not hasattr(chunk, "metadata") or not isinstance(chunk.metadata, dict):
                log.warning(
                    "Warning: Chunk is missing 'metadata' dictionary. Skipping."
                )
                continue

            media_block = chunk.metadata.get("media_block")
            if not media_block:
                log.warning(
                    f"Warning: Chunk with ID {chunk.id if hasattr(chunk, 'id') else 'N/A'} is missing 'media_block' in metadata. Skipping."
                )
                continue

            if not isinstance(media_block, MediaBlock):
                log.warning(
                    f"Warning: 'media_block' in chunk metadata is not a MediaBlock instance for chunk ID {chunk.id if hasattr(chunk, 'id') else 'N/A'}. Skipping."
                )
                continue

            # Validate that it's an image media block
            if media_block.media.media_type != "image":
                log.warning(
                    f"Warning: MediaBlock media_type is '{media_block.media.media_type}', expected 'image' for chunk ID {chunk.id if hasattr(chunk, 'id') else 'N/A'}. Skipping."
                )
                continue

            try:
                # Generate embedding for this single image asynchronously
                embedding_vector = await self.client.a_embed(media_block.media)

                new_embedding = DenseEmbedding(
                    name=self.embedding_name, vector=embedding_vector
                )
                if not hasattr(chunk, "embeddings") or chunk.embeddings is None:
                    chunk.embeddings = []
                    chunk.embeddings.append(new_embedding)

                # Remove the media_block from metadata after successful embedding generation
                # since it's no longer needed and can't be serialized to vectorstore
                chunk.metadata.pop("media_block", None)

            except (
                OSError,
                ValueError,
                FileNotFoundError,
                AttributeError,
                NotImplementedError,
            ) as e:
                # These errors are more likely to be raised by client.a_embed or _create_data_uri
                log.error(
                    f"Error processing image chunk {chunk.id if hasattr(chunk, 'id') else 'N/A'}: {e}. Skipping."
                )
                raise e
            except Exception as e:
                log.error(
                    f"Unexpected error during embedding generation for chunk {chunk.id if hasattr(chunk, 'id') else 'N/A'}: {e}. Skipping."
                )
                raise e

        return processed_chunks
