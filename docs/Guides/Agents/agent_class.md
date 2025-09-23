# Agent

The `Agent` class is the core component for creating autonomous AI agents in DataPizzaAI. It handles task execution, tool management, memory, and planning.

## Basic Usage

```python
from datapizza.agents import Agent
from datapizza.clients import Client
from datapizza.tools import Tool
from datapizza.memory import Memory

agent = Agent(
    name="my_agent",
    system_prompt="You are a helpful assistant",
    client=client,
    # tools=[],
    # max_steps=10,
    # terminate_on_text=True,  # Terminate execution when the client return a plain text 
    # memory=memory,
    # stream=False,
    # planning_interval=0
)

res = agent.run("Hi")
```


## Use Tools

The above agent is quite basic, so let's make it more functional by adding [**tools**](../Tool/tool.md).

```python
from datapizza.tools import tool

@tool
def get_weather(location: str, when: str) -> str:
    """Retrieves weather information for a specified location and time."""
    return "25 °C"

agent = Agent(name"weather_agent",tools=[get_weather], terminate_on_text=True)
response = agent.run("What's the weather tomorrow in Milan?")

print(response)
# Output:
# Tomorrow in Milan, the temperature will be 25 °C.
```


### tool_choice

You can set the parameter `tool_choice` at invoke time.

The accepted values are :  `auto`, `required`, `none`, `required_first`,  `list["tool_name"]`


```python
res = master_agent.run(
    task_input="what is the weather in milan?", tool_choice="required_first"
)
```

- `auto` : the model will decide if use a tool or not.
- `required_first` : force to use a tool only at the first step, then auto.
- `required` : force to use a tool at every step.
- `none` : force to not use any tool.



## Core Methods


### Sync run

`run(task_input: str, tool_choice = "auto", **kwargs) -> str`
Execute a task and return the final result.

```python
result = agent.run("What's the weather like today?")
print(result)  # "The weather is sunny with 25°C"
```

### Stream invoke
`stream_invoke(task_input: str, tool_choice = "auto", **kwargs) -> Generator[str | StepAction, None, None]`
Stream the agent's execution process, yielding intermediate steps. (Do not stream the single answer)

```python
for step in agent.stream_invoke("Analyze this data"):
    if isinstance(step, str):
        print(f"Final: {step}")
    elif isinstance(step, StepAction):
        print(f"Step {step.index} starting...")
```

### Async run

`a_run(task_input: str, **kwargs) -> str`
Async version of run.

```python

async def main():
    result = await agent.a_run("Process this request")

asyncio.run(main())
```

### Async stream invoke
`a_stream_invoke(task_input: str, **kwargs) -> AsyncGenerator[str | StepAction, None]`
Stream the agent's execution process, yielding intermediate steps. (Do not stream the single answer)

```python
async def get_response():
    async for chunk in client.a_stream_invoke("tell me a joke"):
        if isinstance(step, str):
                print(f"Final: {step}")
        elif isinstance(step, StepAction):
            print(f"Step {step.index} starting...")


asyncio.run(get_response())
```


## Multi-Agent Communication

An agent can call another ones using `can_call` method


```python
weather_agent = Agent(name="weather", client=client)
data_agent = Agent(name="data", client=client)

# Allow data_agent to call weather_agent
data_agent.can_call(weather_agent)
```

Alternatively, you can define a tool that manually calls the agent.
The two solutions are more or less identical.

```python
from datapizza.tools import tool
from datapizza.agent import Agent

class MyCustomAgent(Agent):
    system_prompt = "You are an helpfull assistant"

    @tool
    def call_agent_1():
        weather_agent = Agent()
        task_to_ask = something()
        res = weather_agent.run(task_to_ask)

        return res
    
    # Prefer Async tools when possible
    @tool
    async def call_agent_2():
        weather_agent = Agent()
        task_to_ask = something()
        res = await weather_agent.a_run(task_to_ask)

        return res
    
```



## Planning System

When `planning_interval > 0`, the agent creates execution plans at regular intervals:

During the planning stages, the agent spends time thinking about what the next steps are to be taken to achieve the task.

```python
agent = Agent(
    client=client,
    planning_interval=3,  # Plan every 3 steps
)
```

The planning system generates structured plans that help the agent organize complex tasks.


## Stream output


> **_WARNING:_**  This is a beta feature, it does not work with funcion calls (yet)

```python

from datapizza.agents import Agent
from datapizza.clients import ClientResponse, OpenAIClient
from datapizza.tools import tool

client = OpenAIClient(api_key="YOUR_API_KEY", model="gpt-4")

agent = Agent(
    name="Big_boss",
    client=client,
    system_prompt="You are a helpful assistant that answers questions based on the provided context.",
    stream=True,
)

for r in agent.stream_invoke("What is the weather in Milan?"):
    if isinstance(r, ClientResponse):
        print(r.delta, end="", flush=True)
```
