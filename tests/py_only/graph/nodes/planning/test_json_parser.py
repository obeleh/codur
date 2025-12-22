from codur.graph.nodes.planning.json_parser import JSONResponseParser, clean_json_response


def test_clean_json_response_extracts_object() -> None:
    content = 'Here is result: {"ok": true} trailing'
    assert clean_json_response(content) == '{"ok": true}'


def test_json_response_parser_fallback_regex() -> None:
    parser = JSONResponseParser()
    content = "noise {\"value\": 42} more"
    assert parser.parse(content) == {"value": 42}
