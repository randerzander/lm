import os
import json
import unittest
from process_pdf import extract


class TestMultimodalExtraction(unittest.TestCase):
    """Test case for multimodal_test.pdf extraction results."""
    
    def setUp(self):
        """Set up test by processing the multimodal_test.pdf file."""
        # Process the multimodal_test.pdf file to generate extraction results
        self.extract_dir = "extracts/multimodal_test"
        self.result = extract(pdf_path="data/multimodal_test.pdf", extract_dir=self.extract_dir)
        
        # Load the extracted content to analyze
        self.extracted_content_path = os.path.join(self.extract_dir, "extracted_content.json")
        if os.path.exists(self.extracted_content_path):
            with open(self.extracted_content_path, 'r') as f:
                self.content = json.load(f)
        else:
            # If extracted_content.json doesn't exist, get content directly from result
            self.content = self.result

    def test_page_counts(self):
        """Test that the document has exactly 3 pages."""
        pages = self.content.get('pages', {})
        self.assertEqual(len(pages), 3, f"Expected 3 pages, but got {len(pages)}")
        
        # Verify page names are as expected (assuming page_001, page_002, page_003)
        expected_pages = ['page_001', 'page_002', 'page_003']
        actual_pages = sorted(pages.keys())
        self.assertEqual(actual_pages, expected_pages, f"Expected pages {expected_pages}, but got {actual_pages}")

    def test_page_1_elements(self):
        """Test that page 1 has 1 table and 1 chart."""
        page_1_content = self.content['pages'].get('page_001', {})
        elements = page_1_content.get('elements', [])
        
        # Count tables and charts
        table_count = sum(1 for elem in elements if elem.get('type') == 'table')
        chart_count = sum(1 for elem in elements if elem.get('type') == 'chart')
        
        self.assertEqual(table_count, 1, f"Page 1 should have 1 table, but has {table_count}")
        self.assertEqual(chart_count, 1, f"Page 1 should have 1 chart, but has {chart_count}")

    def test_page_2_elements(self):
        """Test that page 2 has 1 table and 1 chart."""
        page_2_content = self.content['pages'].get('page_002', {})
        elements = page_2_content.get('elements', [])
        
        # Count tables and charts
        table_count = sum(1 for elem in elements if elem.get('type') == 'table')
        chart_count = sum(1 for elem in elements if elem.get('type') == 'chart')
        
        self.assertEqual(table_count, 1, f"Page 2 should have 1 table, but has {table_count}")
        self.assertEqual(chart_count, 1, f"Page 2 should have 1 chart, but has {chart_count}")

    def test_page_3_elements(self):
        """Test that page 3 has 1 chart."""
        page_3_content = self.content['pages'].get('page_003', {})
        elements = page_3_content.get('elements', [])
        
        # Count charts
        chart_count = sum(1 for elem in elements if elem.get('type') == 'chart')
        
        self.assertEqual(chart_count, 1, f"Page 3 should have 1 chart, but has {chart_count}")
        
        # Also check that there are no tables on page 3
        table_count = sum(1 for elem in elements if elem.get('type') == 'table')
        self.assertEqual(table_count, 0, f"Page 3 should have 0 tables, but has {table_count}")

    def test_total_element_counts(self):
        """Test the total counts of tables and charts across all pages."""
        all_elements = []
        for page_name, page_data in self.content['pages'].items():
            all_elements.extend(page_data.get('elements', []))
        
        total_tables = sum(1 for elem in all_elements if elem.get('type') == 'table')
        total_charts = sum(1 for elem in all_elements if elem.get('type') == 'chart')
        
        self.assertEqual(total_tables, 2, f"Document should have 2 tables total, but has {total_tables}")
        self.assertEqual(total_charts, 3, f"Document should have 3 charts total, but has {total_charts}")


if __name__ == '__main__':
    # Ensure NVIDIA_API_KEY is set before running tests
    if not os.getenv('NVIDIA_API_KEY'):
        print("Error: NVIDIA_API_KEY environment variable not set. Please set it before running tests.")
        exit(1)
    
    # Run the test
    unittest.main()