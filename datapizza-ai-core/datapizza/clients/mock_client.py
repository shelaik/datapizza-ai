import asyncio
import logging
from typing import Literal

from datapizza.core.clients.client import Client
from datapizza.core.clients.response import ClientResponse
from datapizza.memory.memory import Memory
from datapizza.tools.tools import Tool
from datapizza.type import (
    FunctionCallBlock,
    FunctionCallResultBlock,
    Model,
    StructuredBlock,
    TextBlock,
)

log = logging.getLogger(__name__)


class MockClient(Client):
    """A client for interacting with the Mock API.

    This class provides methods for invoking the Mock API to generate responses
    based on given input data. It extends the InferenceClient class.
    """

    def __init__(
        self,
        model_name: str | None = None,
        system_prompt: str = "",
        temperature: float = 0.6,
    ):
        super().__init__(
            model_name or "mock_client",
            system_prompt=system_prompt,
            temperature=temperature,
        )

        self.memory_adapter = None

    def _invoke(
        self,
        *,
        input: str,
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.6,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
        **kwargs,
    ):
        if tools is None:
            tools = []

        if isinstance(input, list):
            for b in input:
                if isinstance(b, TextBlock):
                    input = b.content

        if memory and isinstance(memory[-1].blocks[-1], FunctionCallResultBlock):
            return ClientResponse(
                content=[TextBlock(content=memory[-1].blocks[-1].result)]
            )

        if not input:
            return ClientResponse(
                content=[
                    TextBlock(
                        content="hi i got this input: "
                        + " and a memory of length: "
                        + (str(len(memory)) if memory else "None")
                    )
                ]
            )

        if "function" in input and tools:
            arguments = {
                "text": "This is a test",
            }
            return ClientResponse(
                content=[
                    FunctionCallBlock(
                        id="1",
                        arguments=arguments,
                        name=tools[0].name,
                        tool=tools[0],
                    )
                ]
            )

        if "exception" in input:
            raise Exception("This is a test exception")

        if memory:
            text = ""
            for b in memory.iter_blocks():
                text += b.content

            text += input

            return ClientResponse(
                content=[TextBlock(content=text)],
                prompt_tokens_used=len(input),
                completion_tokens_used=len(text),
                cached_tokens_used=0,
            )

        return ClientResponse(
            content=[TextBlock(content=input)],
            prompt_tokens_used=len(input),
            completion_tokens_used=len(input),
            cached_tokens_used=0,
        )

    async def _a_invoke(
        self,
        input: str,
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.6,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
        **kwargs,
    ):
        if tools is None:
            tools = []
        return self._invoke(
            input=input,
            tools=tools,
            memory=memory,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            **kwargs,
        )

    def _structured_response(
        self,
        input: list[TextBlock],
        output_cls: type[Model],
        memory: Memory | None = None,
        temperature: float = 0.6,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ):
        # return output_cls.model_validate_json(input)

        return ClientResponse(
            content=[
                StructuredBlock(
                    content=output_cls.model_validate_json(input[0].content)
                )
            ]
        )

    def _stream_invoke(
        self,
        input: list[TextBlock],
        tools: list[str] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.6,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
        **kwargs,
    ):
        if tools is None:
            tools = []
        given_response = ""
        for char in input[0].content:
            given_response += char
            yield ClientResponse(
                content=[TextBlock(content=given_response)], delta=char
            )

    async def _a_stream_invoke(
        self,
        input: list[TextBlock],
        tools: list[str] | None = None,
        memory: Memory | None = None,
        tool_choice: str = "auto",
        temperature: float = 0.6,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
        **kwargs,
    ):
        if tools is None:
            tools = []
        given_response = ""
        for char in input[0].content:
            await asyncio.sleep(0.01)  # Reduced sleep time for faster testing
            given_response += char
            yield ClientResponse(
                content=[TextBlock(content=given_response)], delta=char
            )

    def _embed(self, text: str, model_name: str | None = None) -> list[float]:
        return [0.0] * 1536

    def _convert_tool_choice(self, tool_choice: str | list[str]) -> dict:
        return {}

    def _a_structured_response(
        self,
        input: list[TextBlock],
        output_cls: type[Model],
        memory: Memory | None = None,
        temperature: float = 0.6,
        max_tokens: int = 1000,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        tool_choice: Literal["auto", "required", "none"] | list[str] = "auto",
        **kwargs,
    ):
        return self._structured_response(
            input=input,
            output_cls=output_cls,
            memory=memory,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )
