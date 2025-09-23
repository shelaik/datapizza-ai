from datapizza.tools.tools import Tool


class ToolConverter:
    @staticmethod
    def to_openai_format(tool: Tool) -> dict:
        return {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.properties,
                "required": tool.required,
            },
        }

    @staticmethod
    def to_google_format(tool: Tool) -> dict:
        """Convert OpenAI tool schema to Google GenAI function declaration format."""
        # Extract parameters and remove OpenAI-specific fields
        parameters = {
            "type": tool.schema["parameters"]["type"],
            "properties": tool.schema["parameters"]["properties"],
            "required": tool.schema["parameters"]["required"],
        }

        return {
            "name": tool.schema["name"],
            "description": tool.schema["description"],
            "parameters": parameters,
        }

    @staticmethod
    def to_mistral_format(tool: Tool) -> dict:
        return {"type": "function", "function": tool.schema}
