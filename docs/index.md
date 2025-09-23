`datapizza-ai` is the first Datapizza framework for AI Development. This project aims to provide a robust and flexible framework that can be used to develop and deploy AI models efficiently.

## Features

- Integration with various AI providers (OpenAI, Google VertexAI)
- Support for embeddings and vector search
- RAG (Retrieval-Augmented Generation) functionality
- Document parsing and splitting
- Advanced logging
- Agents and agentic workflows

## Installation

When using `datapizza-ai` in a private repo, you need to do few extra steps to install it since it's not regularly available on PyPi like other python packages. We are going to show the installation process depending on what you are using as dependency manager on your project.

The required username an passwords can be found in Bitwarden under `pypi-registry`.

### Conda environment with pip
If you are using conda, locate the .venv folder of your environment and add a `pip.conf` file with the following content:
```toml
[install]
extra-index-url = https://<username>:<password>@repository.datapizza.tech/repository/datapizza-pypi/simple
```
This previous step has to be done only once per environment. Then, you can install it like any other python package:

```bash
pip install datapizza-ai
```

### Poetry
If you are using poetry as dependency manager, execute the following commands:

```bash
poetry config repositories.datapizza-pypi https://repository.datapizza.tech/repository/datapizza-pypi/
poetry config http-basic.datapizza-pypi <username> <password>
```

This step has to be done only once per project. Then, you can install it like any other package in poetry:

```bash
poetry add datapizza-ai
```

### UV

Add to pyproject.toml in your project folder:

```python
[[tool.uv.index]]
name = "datapizza-pypi"
url = "https://repository.datapizza.tech/repository/datapizza-pypi/simple"
```

MacOS/Linux: Create .netrc file in home directory:

```sh
machine repository.datapizza.tech
login <username>
password <password>
```

Windows: Create _netrc file in home directory (C:/Users/\<username\>/_netrc)

Alternative: Export environment variables:

```bash
export UV_INDEX_INTERNAL_PROXY_USERNAME=<username>
export UV_INDEX_INTERNAL_PROXY_PASSWORD=<password>
```