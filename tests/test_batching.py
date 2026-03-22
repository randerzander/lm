#!/usr/bin/env python3
import importlib
import os
import sys
import types
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def import_query_module():
    fake_lancedb = types.SimpleNamespace(connect=lambda path: None)
    fake_openai = types.SimpleNamespace(OpenAI=object)

    with mock.patch.dict(sys.modules, {"lancedb": fake_lancedb, "openai": fake_openai}):
        sys.modules.pop("query", None)
        return importlib.import_module("query")


class TestQueryCli(unittest.TestCase):
    def test_main_passes_source_document_filter(self):
        query = import_query_module()

        with mock.patch.object(sys, "argv", ["query.py", "--source-document", "doc1", "what is this?"]):
            with mock.patch.object(query.os.path, "exists", return_value=True):
                with mock.patch.object(query, "query_lancedb", return_value=[
                    {"content": "context", "source_document": "doc1", "page_number": "1"}
                ]) as query_lancedb:
                    with mock.patch.object(query, "query_with_llm", return_value="answer"):
                        result = query.main()

        self.assertIsNone(result)
        query_lancedb.assert_called_once_with("./lancedb", "what is this?", "doc1", 5)


if __name__ == "__main__":
    unittest.main()
