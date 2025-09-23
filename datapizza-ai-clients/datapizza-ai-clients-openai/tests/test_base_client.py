from datapizza.clients.openai import (
    OpenAIClient,
)


def test_client_init():
    client = OpenAIClient(
        model="gpt-4o-mini",
        api_key="test_api_key",
    )
    assert client is not None
