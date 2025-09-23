# Google Tools

Google allows you to use the `google_search` tool to search the web for information. 
This tool can be used to retrieve information from the web.

This is an internal google tool that won't be added to function_calls list of the response.

> **_WARNING:_**  This tool only works with GoogleClient

```python
search_tool = {"google_search": {}}
response = client.invoke("Chi ha vinto wimbledon 2025?", tools=[search_tool])
print(response.text)
```
