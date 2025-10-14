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


def process_pdf_with_paths_multiprocessing(pdf_path, texts_dir, pages_dir, pages_per_process=1, max_processes=None):
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
    
    print(f"Processing {num_pages} pages using {max_processes} processes, {pages_per_process} pages per process...")
    
    if num_pages == 1 or max_processes == 1:
        # If only one page or one process, process sequentially
        pdf = pdfium.PdfDocument(pdf_path)
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            
            # Extract text from the page
            text = page.get_textpage().get_text_bounded()
            
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


def process_pdf_with_paths(pdf_path, texts_dir, pages_dir, pages_per_process=1, max_processes=None):
    """
    Process PDF and save outputs to specified directories.
    
    Args:
        pdf_path (str): Path to the PDF file to process
        texts_dir (str): Directory to save extracted text files
        pages_dir (str): Directory to save extracted page images
        pages_per_process (int): Number of pages to process in each process (default: 1)
        max_processes (int): Maximum number of processes to use (default: None, which uses system CPU count)
    """
    # For benchmarking, we can switch between sequential and multiprocessing
    # Uncomment the line below to use sequential processing instead
    # return process_pdf_with_paths_sequential(pdf_path, texts_dir, pages_dir)
    
    # Use multiprocessing version
    return process_pdf_with_paths_multiprocessing(pdf_path, texts_dir, pages_dir, pages_per_process, max_processes)


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
        text = page.get_textpage().get_text_bounded()
        
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
        
        print(f"Processed page {i+1}: saved text to {text_filename} and image to {image_filename}")
        
        # Close the page to free memory
        page.close()
    
    # Close the document
    pdf.close()
    
    pdf_extraction_time = time.time() - start_time
    print(f"PDF extraction completed in {pdf_extraction_time:.2f} seconds")
    
    return pdf_extraction_time


def extract(pdf_path="data/multimodal_test.pdf", output_dir="page_elements", extract_dir=None, timing=False, ocr_titles=True, pages_per_process=1, max_processes=None):
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
        
    Returns:
        dict: A consolidated result object containing all extracted content
    """
    import shutil
    
    start_time = time.time()
    
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
    
    # Step 1: Extract text and images from PDF
    pdf_extraction_time = process_pdf_with_paths(pdf_path, texts_dir, pages_dir, pages_per_process, max_processes)
    
    # Step 2: Process page images to extract content elements with structure and OCR
    import utils
    timing_data = utils.process_page_images(pages_dir=pages_dir, output_dir=elements_dir, timing=timing, ocr_titles=ocr_titles, batch_processing=True, batch_size=20, pdf_extraction_time=pdf_extraction_time, print_timing_summary=False)
    if timing_data:
        page_elements_time = timing_data['page_elements_time']
        table_structure_time = timing_data['table_structure_time']
        chart_structure_time = timing_data['chart_structure_time']
        ocr_time = timing_data['ocr_time']
        ai_processing_time = timing_data['ai_processing_time']
    else:
        ai_processing_time = 0  # Fallback if timing disabled
    
    # Step 3: Create consolidated result object
    result = get_all_extracted_content(pages_dir=pages_dir, output_dir=elements_dir)
    
    # Generate markdown representation of the document
    source_fn = os.path.splitext(os.path.basename(pdf_path))[0] if pdf_path else None
    utils.save_document_markdown(result, extract_dir=extract_dir, source_fn=source_fn)
    
    # Initialize timing variables for new stages
    embeddings_time = 0
    lancedb_time = 0
    
    # Generate embeddings for the markdown content
    if extract_dir and source_fn:
        markdown_path = os.path.join(extract_dir, f"{source_fn}.md")
        if os.path.exists(markdown_path):
            embedding_results, embeddings_time = utils.generate_embeddings_for_markdown(markdown_path)
            if embedding_results:
                utils.save_embeddings_to_json(embedding_results, extract_dir=extract_dir, source_fn=source_fn)
                
                # Save embeddings to LanceDB for queryable storage
                _, lancedb_time = utils.save_to_lancedb(embedding_results, extract_dir=extract_dir, source_fn=source_fn)
    
    # Generate final comprehensive timing summary at the end
    if timing:
        # Calculate detailed times by calling process_page_images again just to get timing info, with print disabled
        # Actually, let's just call it once with timing and capture detailed results
        # Since the detailed timing is already calculated in process_page_images but suppressed, 
        # I need to run process_page_images one more time or pass timing data differently
        # Better approach: modify process_page_images to return timing data when print_timing_summary=False
        
        # Generate final comprehensive timing summary at the end
        if timing and 'timing_data' in locals() and timing_data:
            total_time = time.time() - start_time
            # Calculate percentages
            pdf_extraction_pct = (pdf_extraction_time / total_time) * 100 if total_time > 0 else 0
            page_elements_pct = (page_elements_time / total_time) * 100 if total_time > 0 else 0
            table_structure_pct = (table_structure_time / total_time) * 100 if total_time > 0 else 0
            chart_structure_pct = (chart_structure_time / total_time) * 100 if total_time > 0 else 0
            ocr_pct = (ocr_time / total_time) * 100 if total_time > 0 else 0
            embeddings_pct = (embeddings_time / total_time) * 100 if total_time > 0 else 0
            lancedb_pct = (lancedb_time / total_time) * 100 if total_time > 0 else 0
            
            # Get OCR task counts for breakdown
            ocr_task_counts = timing_data.get('ocr_task_counts', {'table_cells': 0, 'chart_elements': 0, 'titles': 0})
            total_ocr_tasks = ocr_task_counts['table_cells'] + ocr_task_counts['chart_elements'] + ocr_task_counts['titles']
            
            print(f"Timing Summary:")
            print(f"  PDF Extraction: {pdf_extraction_time:.2f}s ({pdf_extraction_pct:.1f}%)")
            print(f"  Page Elements Inference: {page_elements_time:.2f}s ({page_elements_pct:.1f}%)")
            print(f"  Table Structure: {table_structure_time:.2f}s ({table_structure_pct:.1f}%)")
            print(f"  Chart Structure: {chart_structure_time:.2f}s ({chart_structure_pct:.1f}%)")
            
            # OCR with content type breakdown
            if total_ocr_tasks > 0:
                print(f"  OCR: {ocr_time:.2f}s ({ocr_pct:.1f}%) - breakdown:")
                if ocr_task_counts['titles'] > 0:
                    title_pct = (ocr_task_counts['titles'] / total_ocr_tasks) * 100
                    print(f"    Titles: {ocr_task_counts['titles']} tasks ({title_pct:.1f}%)")
                if ocr_task_counts['table_cells'] > 0:
                    cell_pct = (ocr_task_counts['table_cells'] / total_ocr_tasks) * 100
                    print(f"    Table Cells: {ocr_task_counts['table_cells']} tasks ({cell_pct:.1f}%)")
                if ocr_task_counts['chart_elements'] > 0:
                    chart_pct = (ocr_task_counts['chart_elements'] / total_ocr_tasks) * 100
                    print(f"    Chart Elements: {ocr_task_counts['chart_elements']} tasks ({chart_pct:.1f}%)")
            else:
                print(f"  OCR: {ocr_time:.2f}s ({ocr_pct:.1f}%)")
                
            print(f"  Embedding Generation: {embeddings_time:.2f}s ({embeddings_pct:.1f}%)")
            print(f"  LanceDB Indexing: {lancedb_time:.2f}s ({lancedb_pct:.1f}%)")
            print(f"  Total: {total_time:.2f}s")
        elif timing:
            # Fallback for when timing is disabled
            total_time = time.time() - start_time
            print(f"Overall processing completed in {total_time:.2f} seconds")
    
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
    # Sort content types by count in descending order
    sorted_content_types = sorted(content_type_counts.items(), key=lambda x: x[1], reverse=True)
    content_str = ", ".join([f"{content_type}s: {count}" for content_type, count in sorted_content_types])
    # Get text statistics
    text_stats = get_text_stats(output_dir)
    print(f"{content_str} | Total elements: {total_elements}, Words: {text_stats['text_stats']['words']}, Characters: {text_stats['text_stats']['chars']}, Lines: {text_stats['text_stats']['lines']}")
    
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
    



if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    from os.path import isdir, join
    from glob import glob
    
    target_path = "data/multimodal_test.pdf"  # Default single PDF
    pages_per_process = 1
    max_processes = multiprocessing.cpu_count()  # Default to all available CPUs
    
    # Check command line arguments
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        if isdir(arg1):
            # It's a directory - process all PDFs in this directory
            target_path = arg1
        elif arg1.endswith('.pdf') and ('/' in arg1 or '\\' in arg1):
            # It's a PDF file path
            target_path = arg1
            if len(sys.argv) > 2:
                try:
                    pages_per_process = int(sys.argv[2])
                    if len(sys.argv) > 3:
                        try:
                            max_processes = int(sys.argv[3])
                        except ValueError:
                            print(f"Invalid value for max_processes: {sys.argv[3]}. Using default value.")
                except ValueError:
                    print(f"Invalid value for pages_per_process: {sys.argv[2]}. Using default value of 1.")
        else:
            # It could be pages_per_process or max_processes
            try:
                first_num = int(arg1)
                
                # Second argument could be pages_per_process, max_processes, or path
                if len(sys.argv) > 2:
                    try:
                        second_num = int(sys.argv[2])
                        # Both are numbers: first is pages_per_process, second is max_processes
                        pages_per_process = first_num
                        max_processes = second_num
                        
                        # Third argument could be directory or PDF path
                        if len(sys.argv) > 3:
                            arg3 = sys.argv[3]
                            if isdir(arg3):
                                target_path = arg3
                            elif arg3.endswith('.pdf') and ('/' in arg3 or '\\' in arg3):
                                target_path = arg3
                    except ValueError:
                        # Second argument is not a number, so first is pages_per_process
                        pages_per_process = first_num
                        arg2 = sys.argv[2]
                        if isdir(arg2):
                            # Directory path
                            target_path = arg2
                        elif arg2.endswith('.pdf') and ('/' in arg2 or '\\' in arg2):
                            # PDF file path
                            target_path = arg2
                        else:
                            print(f"Invalid path: {arg2}. Expected directory or PDF file path.")
                            sys.exit(1)
                else:
                    # Only one numeric argument provided - assume it's pages_per_process with default PDF
                    pages_per_process = first_num
            except ValueError:
                print(f"Invalid argument: {arg1}. Expected directory, PDF file path, or numeric value.")
                sys.exit(1)
    
    # Check if target_path is a directory
    if os.path.isdir(target_path):
        print(f"Processing all PDFs in directory: {target_path}")
        pdf_files = glob(os.path.join(target_path, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in directory: {target_path}")
            sys.exit(1)
        
        print(f"Found {len(pdf_files)} PDF files to process: {pdf_files}")
        
        # Two-phase processing for multiple files:
        # Phase 1: PDF extraction (text and images) for ALL files first
        print("\nStarting Phase 1: PDF extraction for all files...")
        extraction_results = []
        
        for i, pdf_file in enumerate(pdf_files):
            print(f"\nPDF extraction {i+1}/{len(pdf_files)}: {pdf_file}")
            
            # Create a unique extract directory for each PDF following the new structure
            pdf_name = os.path.splitext(os.path.basename(pdf_file))[0]
            extract_dir = os.path.join("extracts", pdf_name)
            
            # Do just the PDF extraction part (text and images)
            import shutil
            import utils
            
            # Generate default extract_dir based on the source PDF filename
            source_fn = os.path.splitext(os.path.basename(pdf_file))[0]  # Get filename without extension
            extract_dir = os.path.join("extracts", source_fn)

            # Create extract directory and clean it if it already exists
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)

            # Set up paths relative to extract directory
            pages_dir = os.path.join(extract_dir, "pages")
            texts_dir = os.path.join(extract_dir, "texts")
            elements_dir = os.path.join(extract_dir, "page_elements")  # Default output_dir

            # Create subdirectories in extract directory
            os.makedirs(pages_dir, exist_ok=True)
            os.makedirs(texts_dir, exist_ok=True)

            # Extract text and images from PDF (the CPU-intensive part)
            pdf_extraction_time = process_pdf_with_paths(pdf_path=pdf_file, texts_dir=texts_dir, pages_dir=pages_dir, pages_per_process=pages_per_process, max_processes=max_processes)
            
            # Store extraction results for phase 2
            extraction_results.append({
                'pdf_file': pdf_file,
                'extract_dir': extract_dir,
                'pages_dir': pages_dir,
                'texts_dir': texts_dir,
                'elements_dir': elements_dir,
                'pdf_extraction_time': pdf_extraction_time
            })
            
            print(f"Completed PDF extraction for {pdf_file}")
        
        print(f"\nPhase 1 completed: PDF extraction for all {len(pdf_files)} files done.")
        
        # Phase 2: AI processing for all files
        print("\nStarting Phase 2: AI processing (element detection, OCR, etc.) for all files...")
        
        for i, extraction_result in enumerate(extraction_results):
            pdf_file = extraction_result['pdf_file']
            extract_dir = extraction_result['extract_dir']
            pages_dir = extraction_result['pages_dir']
            elements_dir = extraction_result['elements_dir']
            pdf_extraction_time = extraction_result['pdf_extraction_time']
            
            print(f"\nAI processing {i+1}/{len(extraction_results)}: {pdf_file}")
            
            # Process page images to extract content elements with structure and OCR
            ai_start_time = time.time()
            utils.process_page_images(
                pages_dir=pages_dir, 
                output_dir=elements_dir, 
                timing=True, 
                ocr_titles=True, 
                batch_processing=True, 
                batch_size=20, 
                pdf_extraction_time=pdf_extraction_time
            )
            ai_processing_time = time.time() - ai_start_time

            # Create consolidated result object
            result = get_all_extracted_content(pages_dir=pages_dir, output_dir=elements_dir)

            # Generate markdown representation of the document
            source_fn = os.path.splitext(os.path.basename(pdf_file))[0] if pdf_file else None
            utils.save_document_markdown(result, extract_dir=extract_dir, source_fn=source_fn)

            # Initialize timing variables for new stages
            embeddings_time = 0
            lancedb_time = 0

            # Generate embeddings for the markdown content
            if extract_dir and source_fn:
                markdown_path = os.path.join(extract_dir, f"{source_fn}.md")
                if os.path.exists(markdown_path):
                    embedding_results, embeddings_time = utils.generate_embeddings_for_markdown(markdown_path)
                    if embedding_results:
                        utils.save_embeddings_to_json(embedding_results, extract_dir=extract_dir, source_fn=source_fn)
                        
                        # Save embeddings to LanceDB for queryable storage
                        _, lancedb_time = utils.save_to_lancedb(embedding_results, extract_dir=extract_dir, source_fn=source_fn)

            # Report timing if needed
            total_time = time.time() - (time.time() - ai_start_time - pdf_extraction_time)  # Approximate
            print(f"Overall processing completed for {pdf_file} in unknown seconds (extraction: {pdf_extraction_time:.2f}s, AI: {ai_processing_time:.2f}s)")
            print(f"Breakdown for {pdf_file}:")
            print(f"  PDF Extraction: {pdf_extraction_time:.2f}s")
            print(f"  AI Processing (Elements, Structure, OCR): {ai_processing_time:.2f}s")
            print(f"  Embedding Generation: {embeddings_time:.2f}s")
            print(f"  LanceDB Indexing: {lancedb_time:.2f}s")

            # Print content summary for each PDF
            print(f"\nContent summary for {pdf_file}:")
            print("="*50)
            print_content_summary(f"{extract_dir}/page_elements")
            
            # Print detailed content statistics
            from utils import get_content_counts_with_text_stats
            content_counts = get_content_counts_with_text_stats(f"{extract_dir}/page_elements")
            
            print(f"\nContent Type Breakdown:")
            for content_type, stats in content_counts['content_type_breakdown'].items():
                print(f"  {content_type}:")
                print(f"    Elements: {stats['total_elements']}")
                print(f"    Words: {stats['text_stats']['words']}")
                print(f"    Characters: {stats['text_stats']['chars']}")
                print(f"    Lines: {stats['text_stats']['lines']}")
            
            # Save the result to a JSON file in the extraction directory
            from utils import save_extracted_content_to_json
            save_extracted_content_to_json(result, extract_dir=extract_dir)
            
            print(f"\nProcessing completed for {pdf_file}! Total elements found: {content_counts['total_elements']}")
        
        print(f"\nCompleted processing all {len(pdf_files)} PDF files in {target_path}")
        
    else:
        print(f"Processing {target_path} with {pages_per_process} pages per process")
        
        # Example usage - single PDF file
        # When no extract_dir is specified, the extract function will use the default "extracts/{source_fn}" structure
        result = extract(pdf_path=target_path, timing=True, pages_per_process=pages_per_process, max_processes=max_processes)
        
        # The output will automatically be in the "extracts/{source_fn}" directory
        source_fn = os.path.splitext(os.path.basename(target_path))[0]
        output_dir = os.path.join("extracts", source_fn)
        
        # Print content summary
        print("\n" + "="*50)
        print_content_summary(f"{output_dir}/page_elements")
        
        # Print detailed content statistics
        from utils import get_content_counts_with_text_stats
        content_counts = get_content_counts_with_text_stats(f"{output_dir}/page_elements")
        
        print("Per-Page Breakdown:")
        # Sort pages in ascending order (e.g., page_001, page_002, page_003)
        for page_name, page_stats in sorted(content_counts['pages'].items()):
            content_str = ", ".join([f"{content_type}s: {count}" for content_type, count in page_stats['content_types'].items()])
            print(f"  {page_name}: {content_str} | Words: {page_stats['text_stats']['words']}, Characters: {page_stats['text_stats']['chars']}, Lines: {page_stats['text_stats']['lines']}")
        
        # Save the result to a JSON file in the extraction directory
        from utils import save_extracted_content_to_json
        save_extracted_content_to_json(result, extract_dir=output_dir)
        
        print("\nExtraction completed! Total elements found:", content_counts['total_elements'])