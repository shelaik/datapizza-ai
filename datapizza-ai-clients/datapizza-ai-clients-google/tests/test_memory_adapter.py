import pytest

from datapizza.clients.google.memory_adapter import GoogleMemoryAdapter
from datapizza.memory.memory import Memory
from datapizza.type import ROLE, StructuredBlock, TextBlock


def test_google_memory_to_messages_structured_block():
    memory = Memory()
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(StructuredBlock(content={"key": "value"}))
    messages = GoogleMemoryAdapter().memory_to_messages(memory)
    # Google adapter may serialize as string or dict in "parts"
    assert "key" in str(messages[0]["parts"][0])


def test_google_memory_to_messages_with_system_prompt():
    memory = Memory()
    memory.new_turn(role=ROLE.USER)
    memory.add_to_last_turn(TextBlock(content="Hello!"))
    system_prompt = "You are a helpful assistant."
    messages = GoogleMemoryAdapter().memory_to_messages(
        memory, system_prompt=system_prompt
    )
    assert messages[0]["role"] == "model"
    assert system_prompt in str(messages[0]["parts"])
    assert messages[1]["role"] == "user"


def test_google_memory_to_messages_with_input_str():
    memory = Memory()
    input_str = "What is the weather?"
    messages = GoogleMemoryAdapter().memory_to_messages(memory, input=input_str)
    assert messages[-1]["role"] == "user"
    assert input_str in str(messages[-1]["parts"])


def test_google_memory_to_messages_with_input_block():
    memory = Memory()
    input_block = TextBlock(content="This is a block input.")
    messages = GoogleMemoryAdapter().memory_to_messages(memory, input=input_block)
    assert messages[-1]["role"] == "user"
    assert "block input" in str(messages[-1]["parts"])


def test_google_memory_to_messages_unsupported_input():
    memory = Memory()

    class Dummy:
        pass

    with pytest.raises(ValueError):
        GoogleMemoryAdapter().memory_to_messages(memory, input=Dummy())
