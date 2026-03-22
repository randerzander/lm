#!/usr/bin/env python3
import importlib
import os
import sys
import tempfile
import types
import unittest
from unittest import mock


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def import_process_pdf_module():
    fake_pdfium = types.SimpleNamespace(PdfDocument=object)

    with mock.patch.dict(sys.modules, {"pypdfium2": fake_pdfium}):
        sys.modules.pop("process_pdf", None)
        return importlib.import_module("process_pdf")


class TestExtractControlFlow(unittest.TestCase):
    def test_extract_allows_process_page_images_without_timing_data(self):
        process_pdf = import_process_pdf_module()
        import utils

        result_obj = {"pages": {}, "content_elements": {"tables": [], "charts": [], "titles": [], "other": []}, "total_elements": 0}

        with tempfile.TemporaryDirectory() as tmpdir:
            extract_dir = os.path.join(tmpdir, "extracts", "sample")

            with mock.patch.object(process_pdf, "process_pdf_with_paths", return_value=0.1):
                with mock.patch.object(process_pdf, "get_all_extracted_content", return_value=result_obj):
                    with mock.patch.object(utils, "configure_api_rate_limit"):
                        with mock.patch.object(utils, "process_page_images", return_value=None):
                            with mock.patch.object(utils, "save_document_markdown"):
                                with mock.patch.object(utils, "generate_embeddings_from_result", return_value=([], 0)):
                                    with mock.patch.object(utils, "save_to_lancedb", return_value=(None, 0)):
                                        with mock.patch.object(utils, "save_extracted_content_to_json"):
                                            result = process_pdf.extract(
                                                pdf_path="data/sample.pdf",
                                                extract_dir=extract_dir,
                                                timing=False,
                                            )

        self.assertEqual(result, result_obj)


if __name__ == "__main__":
    unittest.main()
