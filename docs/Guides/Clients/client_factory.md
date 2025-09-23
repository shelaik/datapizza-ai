# Client Factory

The `ClientFactory` provides a convenient way to create client instances for different AI providers with consistent configuration.

Instead of creating clients directly, you can use the `ClientFactory` so that you can easily switch between different providers without having to change much code.

## Usage

```python
from datapizza.clients import ClientFactory
import os

# Create a client instance
client = ClientFactory.create(
    provider="mistral",
    api_key=os.getenv("MISTRAL_API_KEY"),
    model="mistral-large-latest",
    system_prompt="You are a helpful assistant that can answer questions about piadina only in italian.",
)
```

## Parameters

- `provider` (str): The AI provider to use. Supported values:
    - `openai`      
    - `google`     
    - `anthropic`
    - `mistral`    
    - `azure_openai`
- `api_key` (str): The API key for the provider
- `model` (str): The specific model to use
- `system_prompt` (str, optional): System-level instructions for the model
- `temperature` (float, optional): Controls randomness in responses (0.0 to 1.0)
- `max_tokens` (int, optional): Maximum number of tokens in the response
- `kwargs` (any)

## Example with Different Providers

```python
# OpenAI client
openai_client = ClientFactory.create(
    provider="openai",
    api_key="your_openai_key",
    model="gpt-4-turbo-preview"
)

# Google client
google_client = ClientFactory.create(
    provider="google",
    api_key="your_google_api_key",
    model="gemini-pro"
)

# Anthropic client
anthropic_client = ClientFactory.create(
    provider="anthropic",
    api_key="api_key",
    model="claude-3-opus-20240229"
)
```

## Notes

- The factory automatically handles provider-specific configurations
- All created clients implement the standard client interface
