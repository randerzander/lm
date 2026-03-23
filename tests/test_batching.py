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

    def test_query_lancedb_uses_hybrid_search_builder(self):
        query = import_query_module()

        fake_search = mock.Mock()
        fake_search.vector.return_value = fake_search
        fake_search.text.return_value = fake_search
        fake_search.rerank.return_value = fake_search
        fake_search.limit.return_value = fake_search
        fake_search.to_list.return_value = [{"content": "result"}]

        fake_table = mock.Mock()
        fake_table.search.return_value = fake_search

        fake_db = mock.Mock()
        fake_db.table_names.return_value = ["all_documents"]
        fake_db.open_table.return_value = fake_table

        fake_embedding_response = mock.Mock()
        fake_embedding_response.data = [mock.Mock(embedding=[0.1, 0.2, 0.3])]

        fake_client = mock.Mock()
        fake_client.embeddings.create.return_value = fake_embedding_response

        with mock.patch.object(query.lancedb, "connect", return_value=fake_db):
            with mock.patch.object(query, "OpenAI", return_value=fake_client):
                results = query.query_lancedb("./lancedb", "hybrid query", None, 5)

        self.assertEqual(results, [{"content": "result"}])
        fake_table.search.assert_called_once_with(query_type="hybrid")
        fake_search.vector.assert_called_once_with([0.1, 0.2, 0.3])
        fake_search.text.assert_called_once_with("hybrid query")
        fake_search.limit.assert_called_once_with(5)


if __name__ == "__main__":
    unittest.main()
