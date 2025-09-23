from datapizza.modules.splitters.text_splitter import TextSplitter


def test_text_splitter():
    text_splitter = TextSplitter(max_char=10, overlap=0)
    chunks = text_splitter.run("This is a test string")
    assert len(chunks) == 3
    assert chunks[0].text == "This is a "
    assert chunks[1].text == "test strin"
    assert chunks[2].text == "g"


def test_text_splitter_with_overlap():
    text_splitter = TextSplitter(max_char=10, overlap=2)
    chunks = text_splitter.run("This is a test string")
    assert len(chunks) == 3

    assert chunks[0].text == "This is a "
    assert chunks[1].text == "a test str"
    assert chunks[2].text == "tring"
