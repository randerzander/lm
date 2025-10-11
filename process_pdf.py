import os
import time
import pypdfium2 as pdfium
from PIL import Image
from utils import get_all_extracted_content
import multiprocessing
from functools import partial


def process_pages_batch(pdf_path, page_nums, texts_dir, pages_dir):
    """
    Process a batch of PDF pages in a separate process.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_nums (list): List of page numbers to process (0-indexed)
        texts_dir (str): Directory to save extracted text files
        pages_dir (str): Directory to save extracted page images
    
    Returns:
        list: List of status messages indicating success or failure for each page
    """
    results = []
    try:
        # Load the PDF document once for this batch
        pdf = pdfium.PdfDocument(pdf_path)
        
        # Process each page in the batch
        for page_num in page_nums:
            try:
                page = pdf.get_page(page_num)
                
                # Extract text from the page
                text = page.get_textpage().get_text_range()
                
                # Save the text to a file
                text_filename = os.path.join(texts_dir, f"page_{page_num+1:03d}.txt")
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
                image_filename = os.path.join(pages_dir, f"page_{page_num+1:03d}.jpg")
                pil_image.save(image_filename, "JPEG", quality=80)  # Lower quality to reduce size
                
                # Close the page to free memory
                page.close()
                
                results.append(f"Processed page {page_num+1}: saved text to {text_filename} and image to {image_filename}")
            except Exception as e:
                results.append(f"Error processing page {page_num+1}: {str(e)}")
        
        # Close the document
        pdf.close()
        
    except Exception as e:
        results.append(f"Error loading PDF {pdf_path}: {str(e)}")
    
    return results


def process_pdf_with_paths_multiprocessing(pdf_path, texts_dir, pages_dir, pages_per_process=1):
    """
    Process PDF and save outputs to specified directories using multiprocessing.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        texts_dir (str): Directory to save extracted text files
        pages_dir (str): Directory to save extracted page images
        pages_per_process (int): Number of pages to process in each process (default: 1)
    """
    # Track time for PDF extraction
    start_time = time.time()
    
    # Create output directories
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(pages_dir, exist_ok=True)
    
    # Load the PDF document to get page count
    pdf = pdfium.PdfDocument(pdf_path)
    num_pages = len(pdf)
    pdf.close()
    
    # Use multiprocessing to process pages in parallel
    # Limit the number of processes to avoid overwhelming the system
    max_processes = min(multiprocessing.cpu_count(), (num_pages + pages_per_process - 1) // pages_per_process, 4)  # Cap at 4 processes
    
    print(f"Processing {num_pages} pages using {max_processes} processes, {pages_per_process} pages per process...")
    
    if num_pages == 1 or max_processes == 1:
        # If only one page or one process, process sequentially
        pdf = pdfium.PdfDocument(pdf_path)
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
    else:
        # Use multiprocessing for multiple pages
        with multiprocessing.Pool(processes=max_processes) as pool:
            # Create batches of pages
            page_batches = []
            for i in range(0, num_pages, pages_per_process):
                batch = list(range(i, min(i + pages_per_process, num_pages)))
                page_batches.append(batch)
            
            # Create arguments for each batch
            args = [(pdf_path, batch, texts_dir, pages_dir) for batch in page_batches]
            
            # Process all batches in parallel
            batch_results = pool.starmap(process_pages_batch, args)
            
            # Print results
            for batch_result in batch_results:
                for result in batch_result:
                    print(result)
    
    pdf_extraction_time = time.time() - start_time
    print(f"PDF extraction completed in {pdf_extraction_time:.2f} seconds")
    return pdf_extraction_time


def process_pdf_with_paths(pdf_path, texts_dir, pages_dir, pages_per_process=1):
    """
    Process PDF and save outputs to specified directories.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        texts_dir (str): Directory to save extracted text files
        pages_dir (str): Directory to save extracted page images
        pages_per_process (int): Number of pages to process in each process (default: 1)
    """
    # For benchmarking, we can switch between sequential and multiprocessing
    # Uncomment the line below to use sequential processing instead
    # return process_pdf_with_paths_sequential(pdf_path, texts_dir, pages_dir)
    
    # Use multiprocessing version
    return process_pdf_with_paths_multiprocessing(pdf_path, texts_dir, pages_dir, pages_per_process)


def process_pdf_with_paths_sequential(pdf_path, texts_dir, pages_dir):
    """
    Process PDF and save outputs to specified directories using sequential processing.
    
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


def extract(pdf_path="data/multimodal_test.pdf", output_dir="page_elements", scratch_dir="scratch", timing=False, ocr_titles=False, pages_per_process=1):
    """
    Complete extraction function that processes a PDF and returns a consolidated result object
    containing all extracted content with texts, filepaths to related images on disk, and bounding boxes.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        output_dir (str): Output directory for extracted elements (relative to scratch_dir)
        scratch_dir (str): Base directory for temporary files, defaults to 'scratch' in current working directory
        timing (bool): Whether to track and report timing for each stage
        ocr_titles (bool): Whether to perform OCR on title elements, defaults to False
        pages_per_process (int): Number of pages to process in each process (default: 1)
        
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
    pdf_extraction_time = process_pdf_with_paths(pdf_path, texts_dir, pages_dir, pages_per_process)
    
    # Step 2: Process page images to extract content elements with structure and OCR
    import utils
    utils.process_page_images(pages_dir=pages_dir, output_dir=elements_dir, timing=timing, ocr_titles=ocr_titles, batch_processing=True, batch_size=5)
    
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


def get_text_stats(output_dir="page_elements"):
    """
    Get text statistics (words, characters, lines) from OCR results.
    
    Args:
        output_dir (str): Output directory where extracted elements are stored
        
    Returns:
        dict: Dictionary containing text statistics
    """
    import glob
    import json
    
    # Initialize counters
    total_words = 0
    total_chars = 0
    total_lines = 0
    total_elements = 0
    total_inference_requests = 0
    
    # Process each content type directory
    for content_type_dir in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isdir(content_type_dir):
            content_type = os.path.basename(content_type_dir)
            
            # Process each JSONL file for this content type
            for jsonl_file in glob.glob(os.path.join(content_type_dir, "*.jsonl")):
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            total_elements += 1
                            data = json.loads(line)
                            
                            # Count OCR requests for this element
                            if content_type in ['table', 'chart']:
                                # Tables and charts have cell-level OCR
                                cells_dir = data['sub_image_path'].replace('.jpg', '_cells')
                                if os.path.exists(cells_dir):
                                    total_inference_requests += len(glob.glob(os.path.join(cells_dir, "*_ocr.json")))
                            elif content_type == 'title' and 'ocr_path' in data:
                                # Titles have direct OCR
                                total_inference_requests += 1
                            elif content_type == 'chart':
                                # Charts have element-level OCR
                                elements_dir = data['sub_image_path'].replace('.jpg', '_elements')
                                if os.path.exists(elements_dir):
                                    total_inference_requests += len(glob.glob(os.path.join(elements_dir, "*_ocr.json")))
                            else:
                                # Other content types (likely figures, equations, etc.)
                                total_inference_requests += 1
                            
                            # Collect text statistics from OCR results if available
                            if content_type == 'table':
                                # For tables, collect text from cell OCR results
                                cells_dir = data['sub_image_path'].replace('.jpg', '_cells')
                                if os.path.exists(cells_dir):
                                    for ocr_file in glob.glob(os.path.join(cells_dir, "*_ocr.json")):
                                        with open(ocr_file, 'r') as ocr_f:
                                            ocr_data = json.load(ocr_f)
                                            if 'data' in ocr_data and ocr_data['data']:
                                                for item in ocr_data['data']:
                                                    if 'text_detections' in item:
                                                        for text_det in item['text_detections']:
                                                            text = text_det['text_prediction']['text']
                                                            total_words += len(text.split())
                                                            total_chars += len(text)
                                                            total_lines += text.count('\n') + 1
                            elif content_type == 'chart':
                                # For charts, collect text from element OCR results
                                elements_dir = data['sub_image_path'].replace('.jpg', '_elements')
                                if os.path.exists(elements_dir):
                                    for ocr_file in glob.glob(os.path.join(elements_dir, "*_ocr.json")):
                                        with open(ocr_file, 'r') as ocr_f:
                                            ocr_data = json.load(ocr_f)
                                            if 'data' in ocr_data and ocr_data['data']:
                                                for item in ocr_data['data']:
                                                    if 'text_detections' in item:
                                                        for text_det in item['text_detections']:
                                                            text = text_det['text_prediction']['text']
                                                            total_words += len(text.split())
                                                            total_chars += len(text)
                                                            total_lines += text.count('\n') + 1
                            elif content_type == 'title' and 'ocr_path' in data:
                                # For titles, collect text from direct OCR result
                                ocr_path = data['ocr_path']
                                if os.path.exists(ocr_path):
                                    with open(ocr_path, 'r') as ocr_f:
                                        ocr_data = json.load(ocr_f)
                                        if 'data' in ocr_data and ocr_data['data']:
                                            for item in ocr_data['data']:
                                                if 'text_detections' in item:
                                                    for text_det in item['text_detections']:
                                                        text = text_det['text_prediction']['text']
                                                        total_words += len(text.split())
                                                        total_chars += len(text)
                                                        total_lines += text.count('\n') + 1
    
    return {
        'total_elements': total_elements,
        'inference_requests': total_inference_requests,
        'text_stats': {
            'words': total_words,
            'chars': total_chars,
            'lines': total_lines
        }
    }


def print_content_summary(output_dir="page_elements"):
    """
    Print a summary of content counts and text statistics.
    
    Args:
        output_dir (str): Output directory where extracted elements are stored
    """
    import glob
    import json
    
    # Initialize counters
    content_type_counts = {}
    total_elements = 0
    total_inference_requests = 0
    
    # Process each content type directory
    for content_type_dir in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isdir(content_type_dir):
            content_type = os.path.basename(content_type_dir)
            content_type_counts[content_type] = 0
            
            # Count elements in this content type
            for jsonl_file in glob.glob(os.path.join(content_type_dir, "*.jsonl")):
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            content_type_counts[content_type] += 1
                            total_elements += 1
    
    # Print summary
    print("Content Summary:")
    print("================")
    for content_type, count in content_type_counts.items():
        print(f"{content_type}: {count} elements")
    print(f"\nTotal elements: {total_elements}")
    
    # Get text statistics
    text_stats = get_text_stats(output_dir)
    print(f"\nText Statistics:")
    print("================")
    print(f"Words: {text_stats['text_stats']['words']}")
    print(f"Characters: {text_stats['text_stats']['chars']}")
    print(f"Lines: {text_stats['text_stats']['lines']}")
    
    # Count inference requests
    for content_type_dir in glob.glob(os.path.join(output_dir, "*")):
        if os.path.isdir(content_type_dir):
            content_type = os.path.basename(content_type_dir)
            
            # Count inference requests for this content type
            for jsonl_file in glob.glob(os.path.join(content_type_dir, "*.jsonl")):
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if content_type in ['table', 'chart']:
                                # Tables and charts have cell-level inference
                                if 'sub_image_path' in data:
                                    cells_dir = data['sub_image_path'].replace('.jpg', '_cells')
                                    if os.path.exists(cells_dir):
                                        total_inference_requests += len(glob.glob(os.path.join(cells_dir, "*_ocr.json")))
                            elif content_type == 'title':
                                # Titles have direct OCR inference
                                total_inference_requests += 1
                            elif content_type == 'chart':
                                # Charts have element-level inference
                                if 'sub_image_path' in data:
                                    elements_dir = data['sub_image_path'].replace('.jpg', '_elements')
                                    if os.path.exists(elements_dir):
                                        total_inference_requests += len(glob.glob(os.path.join(elements_dir, "*_ocr.json")))
                            else:
                                # Other content types (likely figures, equations, etc.)
                                total_inference_requests += 1
    
    print(f"\nTotal inference requests: {total_inference_requests}")


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    pdf_path = "data/multimodal_test.pdf"
    pages_per_process = 1
    
    # Check command line arguments
    if len(sys.argv) > 1:
        # First argument could be either pdf_path or pages_per_process
        arg1 = sys.argv[1]
        if arg1.endswith('.pdf') and ('/' in arg1 or '\\' in arg1):
            # It's a PDF file path
            pdf_path = arg1
            if len(sys.argv) > 2:
                try:
                    pages_per_process = int(sys.argv[2])
                except ValueError:
                    print(f"Invalid value for pages_per_process: {sys.argv[2]}. Using default value of 1.")
        else:
            # It's pages_per_process
            try:
                pages_per_process = int(arg1)
            except ValueError:
                print(f"Invalid value for pages_per_process: {arg1}. Using default value of 1.")
            
            # Second argument could be pdf_path
            if len(sys.argv) > 2:
                if sys.argv[2].endswith('.pdf') and ('/' in sys.argv[2] or '\\' in sys.argv[2]):
                    pdf_path = sys.argv[2]
    
    print(f"Processing {pdf_path} with {pages_per_process} pages per process")
    
    # Example usage
    result = extract(pdf_path=pdf_path, scratch_dir="scratch", timing=True, ocr_titles=False, pages_per_process=pages_per_process)
    
    # Print content summary
    print("\n" + "="*50)
    print_content_summary("scratch/page_elements")
    
    # Print detailed content statistics
    from utils import get_content_counts_with_text_stats
    content_counts = get_content_counts_with_text_stats("scratch/page_elements")
    
    print(f"\nDetailed Content Statistics:")
    print("="*50)
    print(f"Total Elements: {content_counts['total_elements']}")
    print(f"Total Inference Requests: {content_counts['total_inference_requests']}")
    print(f"Total Words: {content_counts['total_text_stats']['words']}")
    print(f"Total Characters: {content_counts['total_text_stats']['chars']}")
    print(f"Total Lines: {content_counts['total_text_stats']['lines']}")
    
    print(f"\nContent Type Breakdown:")
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
    
    print("\nExtraction completed! Total elements found:", content_counts['total_elements'])