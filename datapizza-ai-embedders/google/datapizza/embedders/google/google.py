from datapizza.core.embedder import BaseEmbedder


class GoogleEmbedder(BaseEmbedder):
    def __init__(self, *, api_key: str, model_name: str | None = None):
        self.api_key = api_key
        self.model_name = model_name

        self.client = None
        self.a_client = None

    def _set_client(self):
        import google.generativeai as genai

        if not self.client:
            genai.configure(api_key=self.api_key)
            self.client = genai

    def _set_a_client(self):
        import google.generativeai as genai

        if not self.a_client:
            genai.configure(api_key=self.api_key)
            self.a_client = genai

    def embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float] | list[list[float]]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        [text] if isinstance(text, str) else text

        client = self._get_client()

        result = client.embed_content(model=model, content=text)

        return result.get("embedding")

    async def a_embed(
        self, text: str | list[str], model_name: str | None = None
    ) -> list[float]:
        model = model_name or self.model_name
        if not model:
            raise ValueError("Model name is required.")

        [text] if isinstance(text, str) else text

        client = self._get_a_client()

        result = await client.embed_content_async(model=model, content=text)

        return result.get("embedding")
