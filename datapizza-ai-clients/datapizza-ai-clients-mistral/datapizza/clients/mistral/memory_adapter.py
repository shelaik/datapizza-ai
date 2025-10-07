import base64
import json

from datapizza.memory.memory import Turn
from datapizza.memory.memory_adapter import MemoryAdapter
from datapizza.type import (
    ROLE,
    FunctionCallBlock,
    FunctionCallResultBlock,
    MediaBlock,
    StructuredBlock,
    TextBlock,
)


class MistralMemoryAdapter(MemoryAdapter):
    def _turn_to_message(self, turn: Turn) -> dict:
        content = []
        tool_calls = []
        tool_call_id = None

        for block in turn:
            block_dict = {}

            match block:
                case TextBlock():
                    block_dict = {"type": "text", "text": block.content}
                case FunctionCallBlock():
                    tool_calls.append(
                        {
                            "id": block.id,
                            "function": {
                                "name": block.name,
                                "arguments": json.dumps(block.arguments),
                            },
                            "type": "function",
                        }
                    )
                case FunctionCallResultBlock():
                    tool_call_id = block.id
                    block_dict = {"type": "text", "text": block.result}
                case StructuredBlock():
                    block_dict = {"type": "text", "text": str(block.content)}
                case MediaBlock():
                    match block.media.media_type:
                        case "image":
                            block_dict = self._process_image_block(block)
                        # case "pdf":
                        #    block_dict = self._process_pdf_block(block)

                        case _:
                            raise NotImplementedError(
                                f"Unsupported media type: {block.media.media_type}, only image are supported"
                            )

            if block_dict:
                content.append(block_dict)

        messages: dict = {
            "role": turn.role.value,
        }

        if content:
            messages["content"] = content

        if tool_calls:
            messages["tool_calls"] = tool_calls

        if tool_call_id:
            messages["tool_call_id"] = tool_call_id

        return messages

    def _text_to_message(self, text: str, role: ROLE) -> dict:
        return {"role": role.value, "content": text}

    def _process_image_block(self, block: MediaBlock) -> dict:
        match block.media.source_type:
            case "url":
                return {
                    "type": "image_url",
                    "image_url": {"url": block.media.source},
                }
            case "base64":
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{block.media.extension};base64,{block.media.source}"
                    },
                }
            case "path":
                with open(block.media.source, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{block.media.extension};base64,{base64_image}"
                    },
                }
            case _:
                raise NotImplementedError(
                    f"Unsupported media source type: {block.media.source_type}, only url, base64, path are supported"
                )
