from ddgs import DDGS

from datapizza.tools import Tool


class DuckDuckGoSearchTool(Tool):
    """
    The DuckDuckGo Search tool.
    It allows you to search the web for the given query.
    """

    def __init__(self):
        """Initializes the DuckDuckGoSearch tool."""
        super().__init__(
            name="duckduckgo_search",
            description="Enables DuckDuckGo Search for grounding model responses.",
            func=self.__call__,
        )

    def _format_results(self, results: list[str]) -> str:
        """Format the results."""
        return "## Search Results\n\n" + "\n\n".join(
            [
                f"[{result['title']}]({result['href']})\n{result['body']}"
                for result in results
            ]
        )

    def __call__(self, query: str) -> list[str]:
        """Invoke the tool."""
        res = self.search(query)
        return self._format_results(res)

    def search(self, query: str) -> list[str]:
        """Search the web for the given query."""
        with DDGS() as ddg:
            results = list(ddg.text(query))
            return results
