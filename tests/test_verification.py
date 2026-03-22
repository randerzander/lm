#!/usr/bin/env python3
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image


ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import utils


class TestProcessingAndCounts(unittest.TestCase):
    def test_process_page_images_handles_multiple_internal_batches(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pages_dir = os.path.join(tmpdir, "pages")
            output_dir = os.path.join(tmpdir, "page_elements")
            os.makedirs(pages_dir, exist_ok=True)

            for idx in range(1, 4):
                image_path = os.path.join(pages_dir, f"page_{idx:03d}.jpg")
                Image.new("RGB", (100, 100), color="white").save(image_path, "JPEG")

            bbox = {"x_min": 0.1, "y_min": 0.1, "x_max": 0.9, "y_max": 0.2, "confidence": 0.99}
            batch_results = [
                {"data": [
                    {"bounding_boxes": {"title": [bbox]}},
                    {"bounding_boxes": {"title": [bbox]}},
                ]},
                {"data": [
                    {"bounding_boxes": {"title": [bbox]}},
                ]},
            ]

            with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "test-key"}):
                with mock.patch.object(utils, "calculate_smart_batch_size", return_value=2):
                    with mock.patch.object(utils, "extract_bounding_boxes_batch", return_value=batch_results):
                        utils.process_page_images(
                            pages_dir=pages_dir,
                            output_dir=output_dir,
                            ocr_titles=False,
                            batch_size=3,
                        )

            for idx in range(1, 4):
                jsonl_path = os.path.join(output_dir, "title", f"page_{idx:03d}_elements.jsonl")
                self.assertTrue(os.path.exists(jsonl_path), jsonl_path)
                with open(jsonl_path, "r", encoding="utf-8") as fh:
                    rows = [json.loads(line) for line in fh if line.strip()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["type"], "title")

    def test_get_content_counts_chart_without_elements_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_dir = os.path.join(tmpdir, "chart")
            os.makedirs(chart_dir, exist_ok=True)
            jsonl_path = os.path.join(chart_dir, "page_001_elements.jsonl")
            sub_image_path = os.path.join(chart_dir, "page_001_element_1_chart.jpg")

            with open(jsonl_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"sub_image_path": sub_image_path}) + "\n")

            counts = utils.get_content_counts_with_text_stats(tmpdir)

            self.assertEqual(counts["total_elements"], 1)
            self.assertEqual(counts["total_inference_requests"], 1)
            self.assertEqual(counts["pages"]["page_001"]["inference_requests"], 1)

    def test_get_content_counts_table_counts_structure_and_ocr_once_each(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            table_dir = os.path.join(tmpdir, "table")
            os.makedirs(table_dir, exist_ok=True)
            jsonl_path = os.path.join(table_dir, "page_001_elements.jsonl")
            sub_image_path = os.path.join(table_dir, "page_001_element_1_table.jpg")
            cells_dir = sub_image_path.replace(".jpg", "_cells")
            os.makedirs(cells_dir, exist_ok=True)

            ocr_payload = {
                "data": [
                    {
                        "text_detections": [
                            {"text_prediction": {"text": "cell", "confidence": 0.9}}
                        ]
                    }
                ]
            }
            for idx in range(1, 3):
                with open(os.path.join(cells_dir, f"page_001_element_1_table_cell_{idx}_ocr.json"), "w", encoding="utf-8") as fh:
                    json.dump(ocr_payload, fh)

            with open(jsonl_path, "w", encoding="utf-8") as fh:
                fh.write(json.dumps({"sub_image_path": sub_image_path}) + "\n")

            counts = utils.get_content_counts_with_text_stats(tmpdir)

            self.assertEqual(counts["total_inference_requests"], 3)
            self.assertEqual(counts["pages"]["page_001"]["inference_requests"], 3)
            self.assertEqual(counts["content_type_breakdown"]["table"]["inference_requests"], 3)


if __name__ == "__main__":
    unittest.main()
