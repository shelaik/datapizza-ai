# Blocks

Blocks are the fundamental units of data representation in `datapizza-ai`. They provide a structured way to handle different types of content in your application.

## Base Block

The `Block` class serves as the *abstract* base class for all block types:

```python
class Block:
    def __init__(self, type: str):
        self.type = type
    
    @abstractmethod
    def __hash__(self) -> int:
        pass
```

All blocks have a `type` attribute and must implement a `__hash__` method.

## TextBlock

A `TextBlock` represents simple text content:

```python
class TextBlock(Block):
    def __init__(self, content: str, type: str = "text"):
        self.content = content
        super().__init__(type)
```

TextBlocks are used for standard text responses and messages.

## FunctionCallBlock

A `FunctionCallBlock` represents a call to a function:

```python
class FunctionCallBlock(Block):
    def __init__(
        self,
        id: str,
        arguments: str,
        name: str,
        tool: Tool,
        type: str = "function",
    ):
        self.id = id
        self.arguments = arguments
        self.name = name
        self.tool = tool
        super().__init__(type)
```

FunctionCallBlocks store the function name, arguments, and associated tool information.

## FunctionCallResultBlock

A `FunctionCallResultBlock` stores the result of a function call:

```python
class FunctionCallResultBlock(Block):
    def __init__(
        self,
        id: str,
        tool: Tool,
        result: str,
        type: str = "function_call_result",
    ):
        self.id = id
        self.tool = tool
        self.result = result
        super().__init__(type)
```

This block is used to capture and store the output of tool executions.

## StructuredBlock

A `StructuredBlock` represents structured data using Pydantic models:

```python
class StructuredBlock(Block):
    def __init__(self, content: Model, type: str = "structured"):
        self.content = content
        super().__init__(type)
```

StructuredBlocks allow for strongly-typed, validated data structures.

## MediaBlock

A `MediaBlock` represents media content like images, videos, or audio:

```python
class MediaBlock(Block):
    def __init__(self, media: Media, type: str = "media"):
        self.media = media
        super().__init__(type)
```

MediaBlocks use the `Media` class to store details about the media:

```python
class Media:
    def __init__(
        self,
        *,
        extension: str | None = None,
        media_type: Literal["image", "video", "audio"],
        source_type: Literal["url", "base64", "path", "pil", "raw"],
        source: Any,
        detail: str = "high",
    ):
        self.extension = extension
        self.media_type = media_type
        self.source_type = source_type
        self.source = source
        self.detail = detail
```
