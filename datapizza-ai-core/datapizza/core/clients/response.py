from pydantic import BaseModel

from datapizza.type import (
    Block,
    FunctionCallBlock,
    StructuredBlock,
    TextBlock,
    ThoughtBlock,
)


class ClientResponse:
    """
    A class for storing the response from a client.
    Contains a list of blocks that can be text, function calls, or structured data,
    maintaining the order in which they were generated.

    Args:
        content (List[Block]): A list of blocks.
        delta (str, optional): The delta of the response. Used for streaming responses.
    """

    def __init__(
        self,
        content: list[Block],
        delta: str | None = None,
        stop_reason: str | None = None,
        prompt_tokens_used: int = 0,
        completion_tokens_used: int = 0,
        cached_tokens_used: int = 0,
    ):
        self.content = content
        self.delta = delta
        self.stop_reason = stop_reason
        self.prompt_tokens_used = prompt_tokens_used
        self.completion_tokens_used = completion_tokens_used
        self.cached_tokens_used = cached_tokens_used or 0

    def __eq__(self, other):
        return isinstance(other, ClientResponse) and self.content == other.content

    @property
    def text(self) -> str:
        """Returns concatenated text from all TextBlocks in order"""
        return "\n".join(
            block.content for block in self.content if isinstance(block, TextBlock)
        )

    @property
    def thoughts(self) -> str:
        """Returns all thoughts in order"""
        return "\n".join(
            block.content for block in self.content if isinstance(block, ThoughtBlock)
        )

    @property
    def first_text(self) -> str | None:
        """Returns the content of the first TextBlock or None"""
        text_block = next(
            (item for item in self.content if isinstance(item, TextBlock)), None
        )
        return text_block.content if text_block else None

    @property
    def function_calls(self) -> list[FunctionCallBlock]:
        """Returns all function calls in order"""
        return [item for item in self.content if isinstance(item, FunctionCallBlock)]

    @property
    def structured_data(self) -> list[BaseModel]:
        """Returns all structured data in order"""
        return [
            item.content for item in self.content if isinstance(item, StructuredBlock)
        ]

    def is_pure_text(self) -> bool:
        """Returns True if response contains only TextBlocks"""
        return all(isinstance(block, TextBlock) for block in self.content)

    def is_pure_function_call(self) -> bool:
        """Returns True if response contains only FunctionCallBlocks"""
        return all(isinstance(block, FunctionCallBlock) for block in self.content)

    def __str__(self) -> str:
        return f"ClientResponse(content={self.content}, delta={self.delta}, stop_reason={self.stop_reason})"
