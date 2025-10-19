# Examples

This file collects small, runnable examples for this fork. PRs (to this fork) with improvements or additional use‑cases are welcome.

---

## 1) Image analysis with an OpenAI‑compatible VLM (LM Studio / others)

Analyze an image using a Vision‑Language Model (VLM) exposed via an **OpenAI‑compatible** HTTP API (e.g., LM Studio, local servers, etc.). The model returns a **structured** response validated by **Pydantic**.

### Prerequisites

* A **VLM** running locally and exposed as an OpenAI‑compatible endpoint (e.g., LM Studio at `http://localhost:1234/v1`).

  * Use a model that supports **image inputs** (the exact ID depends on your runtime; the one below is just an example).
* Python 3.10+
* Installed deps:

  ```bash
  pip install -U datapizza-ai pydantic
  ```

> **Note:** If your server/model does not support images, the call will fail. Make sure you are using a **VLM**, not a pure text‑only model.

### Code

```python
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.type import Media, MediaBlock, TextBlock
import base64
from pydantic import BaseModel
from typing import List

class ImageAnalysis(BaseModel):
    Subject: str
    Lighting: str
    Composition: str
    Mood: str
    Technical: str
    Tags: List[str]

# Configure an OpenAI‑compatible client (LM Studio example)
client = OpenAILikeClient(
    api_key="lm-studio",                 # any string is fine for local servers
    model="google/gemma-3-27b-it",       # EXAMPLE model id; adjust to your runtime
    base_url="http://localhost:1234/v1", # LM Studio default
    temperature=0.1
)

# Read the image as base64 (robust across platforms)
with open("foto_prova.png", "rb") as f:
    b64 = base64.b64encode(f.read()).decode("utf-8")

image = Media(
    media_type="image",
    source_type="base64",
    source=b64,
    extension="png"
)

response = client.structured_response(
    input=[
        TextBlock(content=(
            "Act as an expert image reviewer. "
            "Analyse the image and return a concise, structured critique."
        )),
        MediaBlock(media=image),   # <-- make sure this variable matches the Media object above
    ],
    output_cls=ImageAnalysis,
)

# "structured_data" holds Pydantic‑validated results
analysis = response.structured_data[0]
print(f"Subject\n{analysis.Subject}\n")
print(f"Lighting\n{analysis.Lighting}\n")
print(f"Composition\n{analysis.Composition}\n")
print(f"Mood\n{analysis.Mood}\n")
print(f"Technical\n{analysis.Technical}\n")
print(f"Tags\n{', '.join(analysis.Tags)}\n")
```

### Notes & Tips

* **Model ID**: replace `google/gemma-3-27b-it` with the exact VLM ID available in your local server (LM Studio/Ollama/etc.).
* **Images via path**: you can also pass the image by path instead of base64:

  ```python
  image = Media(media_type="image", source_type="path", source="./foto_prova.png", extension="png")
  ```
* **Prompting**: for more deterministic output, lower `temperature` or add explicit formatting instructions to the prompt.
* **Validation**: if the model returns something that cannot be parsed into `ImageAnalysis`, an exception will be raised; tighten your prompt or relax the schema accordingly.

---

More examples coming soon:

* Text ingestion → embeddings → Qdrant (embedded local) → semantic search
* Multi‑vector collections (dense + sparse)
* Streaming responses with OpenAI‑compatible chat models
