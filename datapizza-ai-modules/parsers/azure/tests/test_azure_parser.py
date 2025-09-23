import json
import os

import pytest

from datapizza.modules.parsers.azure import AzureParser
from datapizza.type import NodeType


@pytest.fixture
def sample_azure_result():
    with open(
        os.path.join(os.path.dirname(__file__), "attention_wikipedia_test.json"),
    ) as f:
        sample_result = json.load(f)
    return sample_result


@pytest.fixture
def azure_parser():
    return AzureParser(
        api_key="dummy_key", endpoint="https://dummy-endpoint", result_type="text"
    )


def test_azure_parser_parse(azure_parser, sample_azure_result):
    # Call the public method instead of internal _parse_json
    sample_file_path = os.path.join(
        os.path.dirname(__file__), "attention_wikipedia_test.pdf"
    )
    result = azure_parser._parse_json(sample_azure_result, file_path=sample_file_path)

    assert result.node_type == NodeType.DOCUMENT
    assert result.children
    assert len(result.content) > 30000

    # check if there is at least one child with node_type == NodeType.PARAGRAPH do recursive search
    def check_paragraph(node):
        if node.node_type == NodeType.PARAGRAPH:
            assert len(node.content) > 0
        for child in node.children:
            check_paragraph(child)

    check_paragraph(result)

    # check if there is at least one child with node_type == NodeType.IMAGE do recursive search
    def check_figure(node):
        if node.node_type == NodeType.FIGURE:
            assert node.media.source is not None
        for child in node.children:
            check_figure(child)

    check_figure(result)
