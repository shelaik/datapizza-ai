# datapizza-ai — shelaik fork

> Personal, **unofficial** fork of [`datapizza-labs/datapizza-ai`](https://github.com/datapizza-labs/datapizza-ai). This fork exists to experiment with a more robust **Qdrant embedded (local file) mode** and a few developer‑experience tweaks. It is **not** affiliated with the upstream project.

## Why this fork?

The upstream vectorstore for Qdrant could accidentally fall back to **remote HTTP mode** when users intended to run **fully local** (file‑backed) storage. This fork ships a patched `QdrantVectorstore` that makes local/embedded usage straightforward and predictable.

### Highlights of the patch

* **Embedded (local file) mode** via `path="<dir>"` (recommended).
* **In‑memory mode** via `local_path=":memory:"` (mapped to `location=":memory:"`).
* **Environment cleanup** when embedded is requested: clears `QDRANT_URL`/`QDRANT_HOST` (and double‑underscore forms) to prevent remote fallback.
* **Backend guard**: verifies the client backend is actually *Local*; otherwise raises a clear error.
* **Version‑aware param mapping**:

  * Disk‑backed local ⇒ always use **`path=...`**.
  * `":memory:"` ⇒ map to **`location=":memory:"`**.
  * Avoid using `location="<dir>"` because many client versions treat non‑`:memory:` values as **remote URL**.
* **Async embedded**: best‑effort init with a warning fallback to sync if not supported by your environment.
* **Clear logging** about detected client version, chosen mode, and parameter used.

> ⚠️ Note on Qdrant client semantics: for many releases, `location=":memory:"` means *embedded in RAM*, while any other `location` string is treated like a **URL** (remote). For disk‑backed embedded storage, **use `path=...`**.

## Quick start

### Install this fork

Local editable install for development:

```bash
pip uninstall datapizza -y
pip install -e .
```

Or install directly from Git (read‑only):

```bash
pip install "git+https://github.com/shelaik/datapizza-ai-shelaik-fork@main"
```

### Minimal embedded example (disk‑backed)

```python
import os, logging
from datapizza.vectorstores.qdrant import QdrantVectorstore
from datapizza.core.vectorstore import VectorConfig, Distance
from datapizza.type import EmbeddingFormat, Chunk

logging.basicConfig(level=logging.INFO)
# Ensure environment does not force remote mode
for k in ("QDRANT_URL", "QDRANT__URL", "QDRANT_HOST", "QDRANT__HOST"):
    os.environ.pop(k, None)

# 1) Create local/embedded vectorstore (file‑backed)
vs = QdrantVectorstore(path="./datapizza_qdrant")
print("is_embedded:", vs.is_embedded)
print("backend:", type(vs.get_client()._client).__name__)  # should contain 'Local'

# 2) Create a collection (1536‑dim dense, cosine distance)
vs.create_collection(
    "my_documents",
    vector_config=[
        VectorConfig(
            name="embedding",
            dimensions=1536,
            format=EmbeddingFormat.DENSE,
            distance=Distance.COSINE,
        )
    ],
)

# 3) Upsert a toy chunk (embedding omitted here for brevity)
#    Normally you would generate embeddings using your embedder of choice and attach them to the Chunk.
#    Example: Chunk(embeddings=[DenseEmbedding(name="embedding", vector=[... 1536 floats ...])])

# 4) Search: compute query embedding → vs.search(...)
```

### In‑memory embedded (RAM)

```python
vs = QdrantVectorstore(local_path=":memory:")  # internally mapped to location=":memory:"
```

### Remote/server mode (unchanged)

```python
vs = QdrantVectorstore(host="localhost", port=6333)
```

## Troubleshooting

* **Backend shows `QdrantRemote` but you expected local**

  * Ensure you used `path=...` (not `location="<dir>"`).
  * Clear env vars that may force remote: `QDRANT_URL`, `QDRANT__URL`, `QDRANT_HOST`, `QDRANT__HOST`.
  * Print the backend to verify:

    ```python
    print(type(vs.get_client()._client).__name__)  # 'QdrantLocal' expected
    ```
* **Dimension mismatch** when inserting points

  * The collection’s `dimensions` must match your embedding model output (e.g., 1536 for `text-embedding-3-small`).
* **Async usage in embedded**

  * Not all environments support async embedded cleanly. This fork will log a warning and keep sync‑only if async init fails.

## Keeping in sync with upstream

This is a personal fork. If you want to pull updates from upstream:

```bash
git remote add upstream https://github.com/datapizza-labs/datapizza-ai.git
git fetch upstream
git checkout main
git merge upstream/main
git push origin main
```

## Contributing

Issues and PRs are welcome **to this fork**. If you want the changes to land upstream, feel free to open a PR from your fork to `datapizza-labs/datapizza-ai` with a concise explanation (motivation, behavior change, and backwards compatibility).

## Disclaimer

This fork is provided **as is**, without warranty of any kind, express or implied. Use at your own risk. The author(s) of this fork and the upstream maintainers will not be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with the software or the use or other dealings in the software. This fork is not affiliated with, endorsed by, or supported by the upstream project.

## License

This repository retains the original **MIT License**. See [LICENSE](./LICENSE).

## Attribution

This work is based on the excellent upstream project:

* [`datapizza-labs/datapizza-ai`](https://github.com/datapizza-labs/datapizza-ai)

Qdrant is a separate project; for more details on client semantics and storage options, see the Qdrant documentation.
