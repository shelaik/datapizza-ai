from datapizza.clients.openai import OpenAIClient

from datapizza.clients import ClientFactory


def test_client_factory_openai():
    client = ClientFactory.create(
        provider="openai",
        api_key="test_api_key",
        model="gpt-3.5-turbo",
        system_prompt="You are a helpful assistant that can answer questions about piadina only in italian.",
    )

    assert client is not None

    assert isinstance(client, OpenAIClient)
