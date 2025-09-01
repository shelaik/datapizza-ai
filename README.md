# datapizza-ai

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://img.shields.io/pypi/v/datapizza-ai.svg)](https://pypi.org/project/datapizza-ai/)

🍕 **Build reliable multi-agent AI apps fast** 🍕

`datapizza-ai` provides clear interfaces and predictable behavior for agents and RAG. End-to-end visibility and reliable orchestration keep engineers in control from PoC to scale

## Key Features

- **Integration with AI Providers**: Seamlessly connect with AI services like OpenAI and Google VertexAI.
- **Complex workflows, minimal code.**: Design, automate, and scale powerful agent workflows without the overhead of boilerplate.
- **Retrieval-Augmented Generation (RAG)**: Enhance AI responses with document retrieval.
- **Faster delivery, easier onboarding for new engineers**: Rebuild a RAG + tools agent without multi-class plumbing; parity with simpler, typed interfaces.
- **Up to 40% less debugging time**: Trace and log every LLM/tool call with inputs/outputs

## Quick Start

To get started with `datapizza-ai`, ensure you have Python `>=3.10.0,<3.13.0` installed.

Install the library using pip:

```bash
pip install datapizza-ai
```

## Simple Example of Agents

Here's a basic example demonstrating how to use agents in `datapizza-ai`:

```python
from datapizza.agents import Agent
from datapizza.clients import OpenAIClient

client = OpenAIClient(api_key="YOUR_API_KEY")

agent = Agent(
    name="datapizza_agent",
    client=client,
)

result = agent.run("What is the best TECH company to work for in Italy?")
print(result)
```

For more examples and detailed documentation, please refer to our [complete documentation](LINK_TO_DOCS).

## Contributing

We welcome contributions! If you'd like to contribute to the project, please read our [contribution guidelines](link-to-guidelines).

## License

This project is distributed under the MIT License. See the `LICENSE` file for more details.
