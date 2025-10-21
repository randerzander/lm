import os
import time
import pypdfium2 as pdfium
from PIL import Image
from utils import get_all_extracted_content, log
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
                text = page.get_textpage().get_text_bounded()
                
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


def process_pdf_with_paths(pdf_path, texts_dir, pages_dir, pages_per_process=1, max_processes=None):
    """
    Process PDF and save outputs to specified directories using multiprocessing.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        texts_dir (str): Directory to save extracted text files
        pages_dir (str): Directory to save extracted page images
        pages_per_process (int): Number of pages to process in each process (default: 1)
        max_processes (int): Maximum number of processes to use (default: None, which uses system CPU count)
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
    if max_processes is None:
        max_processes = min(multiprocessing.cpu_count(), (num_pages + pages_per_process - 1) // pages_per_process, 4)  # Cap at 4 processes
    else:
        # Respect user's choice but cap based on page availability
        max_processes = min(max_processes, (num_pages + pages_per_process - 1) // pages_per_process)
    
    log(f"Processing {num_pages} pages using {max_processes} processes, {pages_per_process} pages per process...")
    
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
    
    pdf_extraction_time = time.time() - start_time
    log(f"PDF extraction completed in {pdf_extraction_time:.2f} seconds", level="ALWAYS")
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
        text = page.get_textpage().get_text_bounded()
        
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
        
        log(f"Processed page {i+1}: saved text to {text_filename} and image to {image_filename}")
        
        # Close the page to free memory
        page.close()
    
    # Close the document
    pdf.close()
    
    pdf_extraction_time = time.time() - start_time
    log(f"PDF extraction completed in {pdf_extraction_time:.2f} seconds", level="ALWAYS")
    
    return pdf_extraction_time


def extract(pdf_path="data/multimodal_test.pdf", output_dir="page_elements", extract_dir=None, timing=False, ocr_titles=True, pages_per_process=1, max_processes=None, max_concurrent_requests=10, requests_per_minute=40, max_workers=None):
    """
    Complete extraction function that processes a PDF and returns a consolidated result object
    containing all extracted content with texts, filepaths to related images on disk, and bounding boxes.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        output_dir (str): Output directory for extracted elements (relative to extract_dir)
        extract_dir (str): Base directory for extraction results, defaults to 'extracts/{source_fn}' where source_fn is the PDF filename without extension
        timing (bool): Whether to track and report timing for each stage
        ocr_titles (bool): Whether to perform OCR on title elements, defaults to True
        pages_per_process (int): Number of pages to process in each process (default: 1)
        max_processes (int): Maximum number of processes to use (default: None, which uses system CPU count)
        max_concurrent_requests (int): Maximum number of concurrent API requests allowed (default: 10)
        requests_per_minute (int): Maximum number of requests allowed per minute (default: 40)
        max_workers (int, optional): Maximum number of workers for thread pools (default: None, uses system CPU count)
        
    Returns:
        dict: A consolidated result object containing all extracted content
    """
    import shutil
    
    start_time = time.time()
    
    # Timing for setup operations
    setup_start = time.time()
    # Generate default extract_dir based on the source PDF filename if not provided
    if extract_dir is None:
        source_fn = os.path.splitext(os.path.basename(pdf_path))[0]  # Get filename without extension
        extract_dir = os.path.join("extracts", source_fn)
    
    # Create extract directory and clean it if it already exists
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    
    # Set up paths relative to extract directory
    pages_dir = os.path.join(extract_dir, "pages")
    texts_dir = os.path.join(extract_dir, "texts")
    elements_dir = os.path.join(extract_dir, output_dir)  # output_dir is relative to extract_dir
    
    # Create subdirectories in extract directory
    os.makedirs(pages_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    setup_time = time.time() - setup_start
    
    # Step 1: Extract text and images from PDF
    pdf_extraction_time = process_pdf_with_paths(pdf_path, texts_dir, pages_dir, pages_per_process, max_processes)
    
    # Step 2: Process page images to extract content elements with structure and OCR
    import_start = time.time()
    import utils
    import_time = time.time() - import_start
    
    # Configure API rate limiting
    rate_limit_start = time.time()
    utils.configure_api_rate_limit(max_concurrent_requests, requests_per_minute, max_workers)
    rate_limit_time = time.time() - rate_limit_start
    
    # Process page images to extract content elements with structure and OCR
    ai_processing_start = time.time()
    timing_data = utils.process_page_images(pages_dir=pages_dir, output_dir=elements_dir, timing=timing, ocr_titles=ocr_titles, batch_processing=True, batch_size=25, pdf_extraction_time=pdf_extraction_time)
    ai_processing_time_total = time.time() - ai_processing_start

    page_elements_time = timing_data['page_elements_time']
    table_structure_time = timing_data['table_structure_time']
    chart_structure_time = timing_data['chart_structure_time']
    ocr_time = timing_data['ocr_time']
    # Use the actual AI processing time (the time spent in utils.process_page_images function)
    ai_processing_time = timing_data['ai_processing_time']
    
    # Step 3: Create consolidated result object
    result_creation_start = time.time()
    result = get_all_extracted_content(pages_dir=pages_dir, output_dir=elements_dir)
    result_creation_time = time.time() - result_creation_start
    
    # Generate markdown representation of the document
    markdown_start = time.time()
    source_fn = os.path.splitext(os.path.basename(pdf_path))[0] if pdf_path else None
    utils.save_document_markdown(result, extract_dir=extract_dir, source_fn=source_fn)
    markdown_generation_time = time.time() - markdown_start
    
    # Initialize timing variables for new stages
    embeddings_time = 0
    lancedb_time = 0
    post_processing_time = 0  # Time for saving results to JSON, etc.
    pre_processing_time = 0  # Minimal pre-processing operations after setup
    
    # Generate embeddings for the markdown content
    embedding_generation_start = time.time()
    if extract_dir and source_fn:
        markdown_path = os.path.join(extract_dir, f"{source_fn}.md")
        # Generate granular embeddings from result object instead of markdown file
        embedding_results, embeddings_time = utils.generate_embeddings_from_result(result)
        if embedding_results:
            _, lancedb_time = utils.save_to_lancedb(embedding_results, extract_dir=extract_dir, source_fn=source_fn)
    embedding_generation_time = time.time() - embedding_generation_start
    
    # Measure post-processing: saving results to JSON file and other final operations
    post_processing_start = time.time()
    from utils import save_extracted_content_to_json
    save_extracted_content_to_json(result, extract_dir=extract_dir)
    post_processing_time = time.time() - post_processing_start
    
    # Calculate time for other processing operations that weren't timed separately
    total_time = time.time() - start_time
    if timing and timing_data:  # Only calculate these if timing is enabled and timing_data exists
        accounted_time = setup_time + import_time + rate_limit_time + pdf_extraction_time + ai_processing_time_total + result_creation_time + markdown_generation_time + embedding_generation_time + lancedb_time + post_processing_time
        remaining_processing_time = max(0, total_time - accounted_time)  # Remaining unaccounted time
    else:
        remaining_processing_time = 0
    
    # Generate final comprehensive timing summary at the end
    if timing and timing_data:
        total_time = time.time() - start_time
        # Calculate percentages
        setup_pct = (setup_time / total_time) * 100 if total_time > 0 else 0
        import_pct = (import_time / total_time) * 100 if total_time > 0 else 0
        rate_limit_pct = (rate_limit_time / total_time) * 100 if total_time > 0 else 0
        pdf_extraction_pct = (pdf_extraction_time / total_time) * 100 if total_time > 0 else 0
        ai_processing_pct = (ai_processing_time_total / total_time) * 100 if total_time > 0 else 0  # Total AI processing time
        result_creation_pct = (result_creation_time / total_time) * 100 if total_time > 0 else 0
        markdown_generation_pct = (markdown_generation_time / total_time) * 100 if total_time > 0 else 0
        embedding_generation_pct = (embedding_generation_time / total_time) * 100 if total_time > 0 else 0
        lancedb_pct = (lancedb_time / total_time) * 100 if total_time > 0 else 0
        post_processing_pct = (post_processing_time / total_time) * 100 if total_time > 0 else 0
        remaining_processing_pct = (remaining_processing_time / total_time) * 100 if total_time > 0 else 0
        
        # Get OCR task counts for breakdown
        ocr_task_counts = timing_data.get('ocr_task_counts', {'table_cells': 0, 'chart_elements': 0, 'titles': 0})
        total_ocr_tasks = ocr_task_counts['table_cells'] + ocr_task_counts['chart_elements'] + ocr_task_counts['titles']
        
        log(f"""
Timing Summary:
Setup & Directory Creation: {setup_time:.2f}s ({setup_pct:.1f}%)
Module Import: {import_time:.2f}s ({import_pct:.1f}%)
Rate Limit Configuration: {rate_limit_time:.2f}s ({rate_limit_pct:.1f}%)
PDF Extraction: {pdf_extraction_time:.2f}s ({pdf_extraction_pct:.1f}%)
AI Processing (Elements, Structure, OCR): {ai_processing_time_total:.2f}s ({ai_processing_pct:.1f}%)
        """, level="ALWAYS")
        
        # OCR with content type breakdown (if we have OCR timing data from AI processing)
        if timing_data and 'ocr_task_counts' in timing_data:
            ocr_task_counts = timing_data.get('ocr_task_counts', {'table_cells': 0, 'chart_elements': 0, 'titles': 0})
            total_ocr_tasks = ocr_task_counts['table_cells'] + ocr_task_counts['chart_elements'] + ocr_task_counts['titles']
            if total_ocr_tasks > 0:
                title_pct = (ocr_task_counts['titles'] / total_ocr_tasks) * 100
                cell_pct = (ocr_task_counts['table_cells'] / total_ocr_tasks) * 100
                chart_pct = (ocr_task_counts['chart_elements'] / total_ocr_tasks) * 100
                log(f"""
OCR Breakdown:
  Titles: {ocr_task_counts['titles']} tasks ({title_pct:.1f}%)
  Table Cells: {ocr_task_counts['table_cells']} tasks ({cell_pct:.1f}%)
  Chart Elements: {ocr_task_counts['chart_elements']} tasks ({chart_pct:.1f}%)
                """, level="ALWAYS")
            
        log(f"""
Result Creation: {result_creation_time:.2f}s ({result_creation_pct:.1f}%)
Markdown Generation: {markdown_generation_time:.2f}s ({markdown_generation_pct:.1f}%)
Embedding Generation: {embedding_generation_time:.2f}s ({embedding_generation_pct:.1f}%)
LanceDB Indexing: {lancedb_time:.2f}s ({lancedb_pct:.1f}%)
Post-Processing: {post_processing_time:.2f}s ({post_processing_pct:.1f}%)
Unaccounted/Overhead: {remaining_processing_time:.2f}s ({remaining_processing_pct:.1f}%)
Total: {total_time:.2f}s
        """, level="ALWAYS")

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


def content_summary(output_dir="page_elements"):
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
    
    # Sort content types by count in descending order
    sorted_content_types = sorted(content_type_counts.items(), key=lambda x: x[1], reverse=True)
    content_str = ", ".join([f"{content_type}s: {count}" for content_type, count in sorted_content_types])
    # Get text statistics
    text_stats = get_text_stats(output_dir)
    return f"{content_str} | Total elements: {total_elements}, Words: {text_stats['text_stats']['words']}, Characters: {text_stats['text_stats']['chars']}, Lines: {text_stats['text_stats']['lines']}"
    
if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    from os.path import isdir, join
    from glob import glob
    
    target_path = "data/multimodal_test.pdf"  # Default single PDF
    pages_per_process = 1
    max_processes = multiprocessing.cpu_count()  # Default to all available CPUs
    max_concurrent_requests = 10  # Default: 10 concurrent requests
    requests_per_minute = 40  # Default: 40 requests per minute
    max_workers = None  # Default: system CPU count
    
    log(f"Processing {target_path} with {pages_per_process} pages per process")
    
    # Example usage - single PDF file
    # When no extract_dir is specified, the extract function will use the default "extracts/{source_fn}" structure
    result = extract(pdf_path=target_path, timing=True, pages_per_process=pages_per_process, max_processes=max_processes, max_concurrent_requests=max_concurrent_requests, requests_per_minute=requests_per_minute, max_workers=max_workers)
    
    # The output will automatically be in the "extracts/{source_fn}" directory
    source_fn = os.path.splitext(os.path.basename(target_path))[0]
    output_dir = os.path.join("extracts", source_fn)
    
    log(content_summary(f"{output_dir}/page_elements"), level="ALWAYS")
    
    # Print detailed content statistics
    from utils import get_content_counts_with_text_stats
    content_counts = get_content_counts_with_text_stats(f"{output_dir}/page_elements")
    
    # Sort pages in ascending order (e.g., page_001, page_002, page_003)
    for page_name, page_stats in sorted(content_counts['pages'].items()):
        content_str = ", ".join([f"{content_type}s: {count}" for content_type, count in page_stats['content_types'].items()])
        log(f"{page_name}: {content_str} | Words: {page_stats['text_stats']['words']}, Characters: {page_stats['text_stats']['chars']}, Lines: {page_stats['text_stats']['lines']}", level="ALWAYS")
    
    # Save the result to a JSON file in the extraction directory
    from utils import save_extracted_content_to_json
    save_extracted_content_to_json(result, extract_dir=output_dir)
