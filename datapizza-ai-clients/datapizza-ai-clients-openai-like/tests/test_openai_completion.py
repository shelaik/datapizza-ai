from datapizza.clients.openai_like import OpenAILikeClient


def test_init():
    client = OpenAILikeClient(
        api_key="test_api_key",
        model="gpt-4o-mini",
        system_prompt="You are a helpful assistant that can answer questions about piadina only in italian.",
    )
    assert client is not None
