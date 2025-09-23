from datapizza.clients.openai_completion import OpenAICompletionClient


def test_init():
    client = OpenAICompletionClient(
        api_key="test_api_key",
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant that can answer questions about piadina only in italian.",
    )
    assert client is not None
