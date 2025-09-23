from datapizza.tools.duckduckgo import DuckDuckGoSearchTool


def test_duckduckgo_search():
    tool = DuckDuckGoSearchTool()

    assert tool.name == "duckduckgo_search"
