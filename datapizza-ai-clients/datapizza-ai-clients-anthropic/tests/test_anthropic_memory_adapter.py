from datapizza.clients.anthropic import AnthropicClient
from datapizza.clients.anthropic.memory_adapter import AnthropicMemoryAdapter
from datapizza.memory.memory import Memory
from datapizza.type import ROLE, TextBlock


def test_init_anthropic_client():
    client = AnthropicClient(api_key="test")
    assert client is not None


def test_anthropic_memory_adapter():
    memory_adapter = AnthropicMemoryAdapter()
    memory = Memory()
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.USER)
    memory.add_turn(blocks=[TextBlock(content="Hello, world!")], role=ROLE.ASSISTANT)

    messages = memory_adapter.memory_to_messages(memory)

    assert messages == [
        {
            "role": "user",
            "content": "Hello, world!",
        },
        {
            "role": "assistant",
            "content": "Hello, world!",
        },
    ]
