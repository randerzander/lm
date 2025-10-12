#!/usr/bin/env python3
"""
Test script to demonstrate the multiprocessing functionality with configurable pages per process.
"""

import os
import time
import pypdfium2 as pdfium
from PIL import Image
import multiprocessing
from utils import get_all_extracted_content


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


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    pages_per_process = 1
    if len(sys.argv) > 1:
        try:
            pages_per_process = int(sys.argv[1])
        except ValueError:
            print(f"Invalid value for pages_per_process: {sys.argv[1]}. Using default value of 1.")
    
    print(f"Using {pages_per_process} pages per process")
    
    # Test the function
    result = process_pdf_with_paths_multiprocessing(
        pdf_path="data/multimodal_test.pdf",
        texts_dir="scratch/texts",
        pages_dir="scratch/pages",
        pages_per_process=pages_per_process
    )
    
    print(f"Test completed with pages_per_process={pages_per_process}")