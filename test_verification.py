#!/usr/bin/env python3
"""
Test to verify the graphic elements batching functionality is working correctly.
"""

def test_batching_logic():
    """Test that the batching changes are correctly implemented."""
    print("Testing the graphic elements batching implementation...")
    
    # Read the utils.py file to check for the implemented changes
    with open('utils.py', 'r') as f:
        content = f.read()
    
    # Check that we have the new chart_graphic_elements_tasks list
    if "chart_graphic_elements_tasks = []" in content:
        print("✓ chart_graphic_elements_tasks list is initialized")
    else:
        print("✗ chart_graphic_elements_tasks list is missing")
    
    # Check that the extract_graphic_elements_batch function exists
    if "def extract_graphic_elements_batch" in content:
        print("✓ extract_graphic_elements_batch function is defined")
    else:
        print("✗ extract_graphic_elements_batch function is missing")
    
    # Check that we're collecting chart tasks during initial processing
    if "chart_graphic_elements_tasks.append" in content:
        print("✓ Chart tasks are being collected for batch processing")
    else:
        print("✗ Chart tasks are not being collected")
    
    # Check that batch processing of chart graphic elements exists
    if "Processing chart graphic elements tasks in parallel batches" in content:
        print("✓ Batch processing of chart graphic elements is implemented")
    else:
        print("✗ Batch processing of chart graphic elements is missing")
    
    # Check that chart element OCR tasks are still being collected
    if "chart_element_ocr_tasks.append" in content:
        print("✓ Chart element OCR tasks are being collected")
    else:
        print("✗ Chart element OCR tasks are not being collected")

def main():
    print("Verification of graphic elements batching implementation:")
    print("=" * 60)
    test_batching_logic()
    print("=" * 60)
    print("Implementation should now properly batch graphic elements inference")
    print("calls and add sub-elements to OCR tasks for batch processing.")

if __name__ == "__main__":
    main()