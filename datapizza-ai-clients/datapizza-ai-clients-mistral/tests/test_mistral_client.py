from datapizza.clients.mistral import MistralClient


def test_init_mistral_client():
    client = MistralClient(api_key="test")
    assert client is not None
