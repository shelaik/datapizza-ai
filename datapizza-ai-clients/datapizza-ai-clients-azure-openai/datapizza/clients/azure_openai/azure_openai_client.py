from datapizza.clients.openai.openai_client import OpenAIClient
from datapizza.core.cache import Cache
from openai import AsyncAzureOpenAI, AzureOpenAI


class AzureOpenAIClient(OpenAIClient):
    def __init__(
        self,
        api_key: str,
        azure_endpoint: str,
        *,
        model: str = "gpt-4o-mini",
        system_prompt: str = "",
        temperature: float | None = None,
        cache: Cache | None = None,
        azure_deployment: str | None = None,
        api_version: str | None = None,
    ):
        self.azure_endpoint = azure_endpoint
        self.azure_deployment = azure_deployment
        self.api_version = api_version

        super().__init__(
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            cache=cache,
        )
        self._set_client()

    def _set_client(self):
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
        )

    def _set_a_client(self):
        self.a_client = AsyncAzureOpenAI(
            api_key=self.api_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
        )
