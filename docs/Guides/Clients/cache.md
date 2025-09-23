# Clients Cache

The caching system allows you to store and retrieve API responses, improving performance and reducing API costs for repeated requests.

## Basic Usage

Here's how to set up caching with Redis:

```python
from datapizza.cache import RedisCache
from datapizza.clients import OpenAIClient

# Initialize Redis cache
cache = RedisCache(
    host="localhost",  # Redis server host
    port=6379,        # Redis server port
    db=0,            # Redis database number
    password=None    # Optional Redis password
)

# Create client with cache
client = OpenAIClient(
    api_key="your-api-key",
    model="gpt-4o-mini",
    cache=cache  # Pass the cache instance
)
```

## Caching Examples

### 1. Basic Text Completion

```python
# First request - will call the API and cache the response
response1 = client.invoke(
    input="What is the capital of France?",
    temperature=0.7
)

# Second request with same parameters - will use cached response
response2 = client.invoke(
    input="What is the capital of France?",
    temperature=0.7
)
```