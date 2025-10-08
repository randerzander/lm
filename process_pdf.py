import os
import time
import pypdfium2 as pdfium
from PIL import Image
from utils import get_all_extracted_content

def process_pdf(pdf_path):
    # Track time for PDF extraction
    start_time = time.time()
    
    # Create output directories
    os.makedirs("texts", exist_ok=True)
    os.makedirs("pages", exist_ok=True)
    
    # Load the PDF document
    pdf = pdfium.PdfDocument(pdf_path)
    
    # Process each page
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        
        # Extract text from the page
        text = page.get_textpage().get_text_range()
        
        # Save the text to a file
        text_filename = f"texts/page_{i+1:03d}.txt"
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Render the page as an image
        pil_image = page.render(
            scale=1  # 1x scaling to keep image size manageable for API
        ).to_pil()
        
        # Resize image if needed to meet API constraints
        # Calculate new dimensions keeping aspect ratio
        max_size = (1024, 1024)  # Reasonable size for API
        pil_image.thumbnail(max_size, Image.LANCZOS)
        
        # Save the image
        image_filename = f"pages/page_{i+1:03d}.jpg"
        pil_image.save(image_filename, "JPEG", quality=80)  # Lower quality to reduce size
        
        print(f"Processed page {i+1}: saved text to {text_filename} and image to {image_filename}")
        
        # Close the page to free memory
        page.close()
    
    # Close the document
    pdf.close()
    
    pdf_extraction_time = time.time() - start_time
    print(f"PDF extraction completed in {pdf_extraction_time:.2f} seconds")
    
    return pdf_extraction_time


    


def extract(pdf_path="data/multimodal_test.pdf", output_dir="page_elements", scratch_dir="scratch", timing=False, ocr_titles=False):
    """
    Complete extraction function that processes a PDF and returns a consolidated result object
    containing all extracted content with texts, filepaths to related images on disk, and bounding boxes.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        output_dir (str): Output directory for extracted elements (relative to scratch_dir)
        scratch_dir (str): Base directory for temporary files, defaults to 'scratch' in current working directory
        timing (bool): Whether to track and report timing for each stage
        ocr_titles (bool): Whether to perform OCR on title elements, defaults to False
        
    Returns:
        dict: A consolidated result object containing all extracted content
    """
    import shutil
    
    start_time = time.time()
    
    # Create scratch directory and clean it if it already exists
    if os.path.exists(scratch_dir):
        shutil.rmtree(scratch_dir)
    os.makedirs(scratch_dir, exist_ok=True)
    
    # Set up paths relative to scratch directory
    pages_dir = os.path.join(scratch_dir, "pages")
    texts_dir = os.path.join(scratch_dir, "texts")
    elements_dir = os.path.join(scratch_dir, output_dir)  # output_dir is relative to scratch_dir
    
    # Create subdirectories in scratch
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    
    # Step 1: Extract text and images from PDF
    pdf_extraction_time = process_pdf_with_paths(pdf_path, texts_dir, pages_dir)
    
    # Step 2: Process page images to extract content elements with structure and OCR
    from utils import process_page_images
    process_page_images(pages_dir=pages_dir, output_dir=elements_dir, timing=timing, ocr_titles=ocr_titles)
    
    # Step 3: Create consolidated result object
    result = get_all_extracted_content(pages_dir=pages_dir, output_dir=elements_dir)
    
    # Report timing if requested
    if timing:
        total_time = time.time() - start_time
        print(f"Overall processing completed in {total_time:.2f} seconds")
        print(f"Breakdown:")
        print(f"  PDF Extraction: {pdf_extraction_time:.2f}s")
        print(f"  AI Processing (Elements, Structure, OCR): {total_time - pdf_extraction_time:.2f}s")
    
    return result


def process_pdf_with_paths(pdf_path, texts_dir, pages_dir):
    """
    Process PDF and save outputs to specified directories.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        texts_dir (str): Directory to save extracted text files
        pages_dir (str): Directory to save extracted page images
    """
    # Track time for PDF extraction
    start_time = time.time()
    
    # Create output directories
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)
    
    # Load the PDF document
    pdf = pdfium.PdfDocument(pdf_path)
    
    # Process each page
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        
        # Extract text from the page
        text = page.get_textpage().get_text_range()
        
        # Save the text to a file
        text_filename = os.path.join(texts_dir, f"page_{i+1:03d}.txt")
        with open(text_filename, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Render the page as an image
        pil_image = page.render(
            scale=1  # 1x scaling to keep image size manageable for API
        ).to_pil()
        
        # Resize image if needed to meet API constraints
        # Calculate new dimensions keeping aspect ratio
        max_size = (1024, 1024)  # Reasonable size for API
        pil_image.thumbnail(max_size, Image.LANCZOS)
        
        # Save the image
        image_filename = os.path.join(pages_dir, f"page_{i+1:03d}.jpg")
        pil_image.save(image_filename, "JPEG", quality=80)  # Lower quality to reduce size
        
        print(f"Processed page {i+1}: saved text to {text_filename} and image to {image_filename}")
        
        # Close the page to free memory
        page.close()
    
    # Close the document
    pdf.close()
    
    pdf_extraction_time = time.time() - start_time
    print(f"PDF extraction completed in {pdf_extraction_time:.2f} seconds")
    
    return pdf_extraction_time


def get_text_stats(output_dir="page_elements"):
    """
    Get text statistics (words, characters, lines) from OCR results.
    
    Args:
        output_dir (str): Output directory where extracted elements are stored
        
    Returns:
        dict: A dictionary containing text statistics
    """
    stats = {
        'words': 0,
        'chars': 0,
        'lines': 0
    }
    
    if not os.path.exists(output_dir):
        return stats
    
    # Process all OCR JSON files in the output directory
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith('_ocr.json'):
                ocr_path = os.path.join(root, file)
                with open(ocr_path, 'r') as f:
                    ocr_data = json.load(f)
                    if 'data' in ocr_data and ocr_data['data']:
                        for ocr_item in ocr_data['data']:
                            if 'text_detections' in ocr_item:
                                for text_det in ocr_item['text_detections']:
                                    text = text_det['text_prediction']['text']
                                    stats['words'] += len(text.split())
                                    stats['chars'] += len(text)
                                    stats['lines'] += text.count('\n') + 1
    
    return stats


if __name__ == "__main__":
    # Run the complete extraction process
    result = extract(scratch_dir="scratch", timing=True, ocr_titles=False)
    
    # Print summary info
    print(f"Extraction completed! Total elements found: {result['total_elements']}")
    print(f"Tables: {len(result['content_elements']['tables'])}")
    print(f"Charts: {len(result['content_elements']['charts'])}")
    print(f"Titles: {len(result['content_elements']['titles'])}")
    
    # Import and use the content counting utility with text stats
    from utils import get_content_counts_with_text_stats
    content_counts = get_content_counts_with_text_stats(output_dir="scratch/page_elements")
    
    print("\nDetailed Content Statistics:")
    print("="*50)
    print(f"Total Elements: {content_counts['total_elements']}")
    print(f"Total Inference Requests: {content_counts['total_inference_requests']}")
    print(f"Total Words: {content_counts['total_text_stats']['words']}")
    print(f"Total Characters: {content_counts['total_text_stats']['chars']}")
    print(f"Total Lines: {content_counts['total_text_stats']['lines']}")
    print()
    
    print("Content Type Breakdown:")
    for content_type, stats in content_counts['content_type_breakdown'].items():
        print(f"  {content_type}:")
        print(f"    Elements: {stats['total_elements']}")
        print(f"    Inference Requests: {stats['inference_requests']}")
        print(f"    Words: {stats['text_stats']['words']}")
        print(f"    Characters: {stats['text_stats']['chars']}")
        print(f"    Lines: {stats['text_stats']['lines']}")
    print()
    
    print("Per-Page Breakdown:")
    for page_name, page_stats in content_counts['pages'].items():
        print(f"  {page_name}:")
        for content_type, count in page_stats['content_types'].items():
            print(f"    {content_type}: {count} elements")
        print(f"    Inference requests: {page_stats['inference_requests']}")
        print(f"    Words: {page_stats['text_stats']['words']}")
        print(f"    Characters: {page_stats['text_stats']['chars']}")
        print(f"    Lines: {page_stats['text_stats']['lines']}")
    
    # Save the result to a JSON file
    from utils import save_extracted_content_to_json
    save_extracted_content_to_json(result)