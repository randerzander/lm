import os
import json
import time
import requests
import base64
from glob import glob
from PIL import Image
import concurrent.futures
from threading import Lock

# Global lock for API requests to prevent overwhelming the server
api_request_lock = Lock()

def _make_batch_request(items, api_endpoint, headers, batch_size, payload_processor, result_processor, api_description="batch", max_workers=None, parallel=False):
    """
    Generic function for making batch requests to an API.
    
    Args:
        items: List of items to process in batches
        api_endpoint: API endpoint URL
        headers: Headers for the API request
        batch_size: Number of items to process in each batch
        payload_processor: Function that takes a batch of items and returns the payload
        result_processor: Function that processes the API response
        api_description: Description for logging purposes
        max_workers: Maximum number of worker threads for parallel processing (default: CPU count)
        parallel: Whether to process batches in parallel (default: False)
        
    Returns:
        list: List of results from processing each batch
    """
    results = []
    
    # Create all batches first
    all_batches = []
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        all_batches.append(batch_items)
    
    if parallel and len(all_batches) > 1:
        # Process all batches in parallel
        print(f"Processing {len(all_batches)} {api_description} batches in parallel...")
        
        def process_single_batch(batch_idx_batch_items):
            batch_idx, batch_items = batch_idx_batch_items
            print(f"Processing {api_description} batch {batch_idx + 1}/{len(all_batches)} ({len(batch_items)} items)")
            print(f"  Items in batch: {[os.path.basename(item) if isinstance(item, str) else item for item in batch_items]}")
            
            # Prepare batch payload using the provided processor
            payload = payload_processor(batch_items)
            
            try:
                # Use lock to prevent too many concurrent requests to the API
                with api_request_lock:
                    response = requests.post(api_endpoint, headers=headers, json=payload)
                
                if response.status_code == 200:
                    batch_result = response.json()
                    processed_result = result_processor(batch_result)
                    return processed_result
                else:
                    print(f"{api_description.title()} API request failed with status {response.status_code}: {response.text}")
                    # Raise exception to be handled by the executor
                    raise requests.exceptions.RequestException(f"{api_description.title()} API request failed: {response.status_code}")
            except Exception as e:
                print(f"Error processing {api_description} batch: {str(e)}")
                raise
        
        # Prepare batch index pairs for processing
        batch_index_pairs = [(i, batch) for i, batch in enumerate(all_batches)]
        
        # Use ThreadPoolExecutor to process all batches in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all batch processing tasks
            future_to_batch = {executor.submit(process_single_batch, batch_idx_pair): batch_idx_pair 
                              for batch_idx_pair in batch_index_pairs}
            
            # Collect results in order of submission to maintain batch sequence
            batch_results = [None] * len(all_batches)
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_idx_pair = future_to_batch[future]
                batch_idx = batch_idx_pair[0]
                try:
                    result = future.result()
                    batch_results[batch_idx] = result
                except Exception as e:
                    print(f"Batch {batch_idx + 1} generated an exception: {str(e)}")
                    raise
            
            # Add all results in order
            results.extend(batch_results)
    else:
        # Process items in batches sequentially (original behavior)
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            print(f"Processing {api_description} batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1} ({len(batch_items)} items)")
            print(f"  Items in batch: {[os.path.basename(item) if isinstance(item, str) else item for item in batch_items]}")
            
            # Prepare batch payload using the provided processor
            payload = payload_processor(batch_items)
            
            try:
                # Use lock to prevent too many concurrent requests to the API
                with api_request_lock:
                    response = requests.post(api_endpoint, headers=headers, json=payload)
                
                if response.status_code == 200:
                    batch_result = response.json()
                    processed_result = result_processor(batch_result)
                    results.append(processed_result)
                else:
                    print(f"{api_description.title()} API request failed with status {response.status_code}: {response.text}")
                    # Return partial results or raise exception based on requirements
                    raise requests.exceptions.RequestException(f"{api_description.title()} API request failed: {response.status_code}")
            except Exception as e:
                print(f"Error processing {api_description} batch: {str(e)}")
                # Continue with other batches or raise exception based on requirements
                raise
    
    return results


def _make_embedding_batch_request(items, client, batch_size, api_description="embedding"):
    """
    Generic function for making batch requests for embeddings.
    
    Args:
        items: List of items (text content) to process in batches
        client: OpenAI client for embeddings
        batch_size: Number of items to process in each batch
        api_description: Description for logging purposes
        
    Returns:
        list: List of results from processing each item
    """
    results = []
    
    # Process items in batches
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        print(f"Processing {api_description} batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1} ({len(batch_items)} items)")
        
        # Prepare batch content
        batch_contents = [item[1] for item in batch_items]  # item[1] is the content
        
        try:
            # Generate embeddings for the batch
            response = client.embeddings.create(
                input=batch_contents,
                model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                encoding_format="float",
                extra_body={"modality": ["text"] * len(batch_contents), "input_type": "query", "truncate": "NONE"}
            )
            
            # Process each embedding in the batch response
            for j, (page_idx, content) in enumerate(batch_items):
                embedding = response.data[j].embedding
                
                results.append({
                    'page_index': page_idx,
                    'content': content,
                    'embedding': embedding
                })
                
                print(f"Generated embedding for page {page_idx} (length: {len(content)} chars)")
                
        except Exception as e:
            print(f"Error processing {api_description} batch starting at page {batch_items[0][0]}: {str(e)}")
            # Fallback: process individually if batch fails
            for page_idx, content in batch_items:
                try:
                    response = client.embeddings.create(
                        input=[content],
                        model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                        encoding_format="float",
                        extra_body={"modality": ["text"], "input_type": "query", "truncate": "NONE"}
                    )
                    
                    embedding = response.data[0].embedding
                    
                    results.append({
                        'page_index': page_idx,
                        'content': content,
                        'embedding': embedding
                    })
                    
                    print(f"Generated embedding for page {page_idx} (length: {len(content)} chars) using fallback")
                    
                except Exception as e_single:
                    print(f"Error generating embedding for page {page_idx}: {str(e_single)}")
                    results.append({
                        'page_index': page_idx,
                        'content': content,
                        'embedding': None,
                        'error': str(e_single)
                    })
    
    return results


def extract_bounding_boxes(image_path, api_key=None):
    """
    Extract bounding boxes for various content types from an image using NVIDIA AI API.
    
    Args:
        image_path (str): Path to the image file
        api_key (str, optional): API key for authorization. If not provided, 
                                assumes running in NGC environment
    
    Returns:
        dict: JSON response containing bounding boxes for various content types
    """
    invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    assert len(image_b64) < 180_000, \
      "To upload larger images, use the assets API (see docs)"

    # Set authorization header
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    else:
        headers = {
            "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
            "Accept": "application/json"
        }

    payload = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:image/png;base64,{image_b64}"
            }
        ]
    }

    print(f"Processing individual page element inference for: {image_path}")
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()


def extract_bounding_boxes_batch(image_paths, api_key=None, batch_size=5):
    """
    Extract bounding boxes for various content types from multiple images using NVIDIA AI API in batches.
    
    Args:
        image_paths (list): List of paths to image files
        api_key (str, optional): API key for authorization. If not provided, 
                                assumes running in NGC environment
        batch_size (int): Number of images to process in each batch (default: 5)
    
    Returns:
        list: List of JSON responses containing bounding boxes for various content types
    """
    invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-page-elements-v2"
    
    # Set authorization header
    # If api_key is None, try to get it from environment variable
    if api_key is None:
        api_key = os.getenv('NVIDIA_API_KEY')
    
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    else:
        headers = {
            "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
            "Accept": "application/json"
        }

    def payload_processor(batch_paths):
        inputs = []
        for img_path in batch_paths:
            with open(img_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            
            assert len(image_b64) < 180_000, \
              f"Image {img_path} is too large. To upload larger images, use the assets API (see docs)"
            
            inputs.append({
                "type": "image_url",
                "url": f"data:image/jpeg;base64,{image_b64}"
            })
        
        return {"input": inputs}

    def result_processor(batch_result):
        return batch_result

    return _make_batch_request(
        image_paths, 
        invoke_url, 
        headers, 
        batch_size, 
        payload_processor, 
        result_processor, 
        "page elements"
    )


def extract_table_structure(image_path, api_key=None):
    """
    Extract table structure from an image using NVIDIA AI API.
    
    Args:
        image_path (str): Path to the image file
        api_key (str, optional): API key for authorization. If not provided, 
                                assumes running in NGC environment
    
    Returns:
        dict: JSON response containing table structure information
    """
    invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    assert len(image_b64) < 180_000, \
      "To upload larger images, use the assets API (see docs)"

    # Set authorization header
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    else:
        headers = {
            "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
            "Accept": "application/json"
        }

    payload = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:image/png;base64,{image_b64}"
            }
        ]
    }

    print(f"Processing individual table structure inference for: {image_path}")
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()


def extract_table_structure_batch(image_paths, api_key=None, batch_size=5):
    """
    Extract table structure from multiple images using NVIDIA AI API in batches.
    
    Args:
        image_paths (list): List of paths to image files
        api_key (str, optional): API key for authorization. If not provided, 
                                assumes running in NGC environment
        batch_size (int): Number of images to process in each batch (default: 5)
    
    Returns:
        list: List of JSON responses containing table structure information for each image
    """
    invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-table-structure-v1"
    
    # Set authorization header
    # If api_key is None, try to get it from environment variable
    if api_key is None:
        api_key = os.getenv('NVIDIA_API_KEY')
    
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    else:
        headers = {
            "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
            "Accept": "application/json"
        }

    def payload_processor(batch_paths):
        inputs = []
        for img_path in batch_paths:
            with open(img_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            
            assert len(image_b64) < 180_000, \
              f"Image {img_path} is too large. To upload larger images, use the assets API (see docs)"
            
            inputs.append({
                "type": "image_url",
                "url": f"data:image/jpeg;base64,{image_b64}"
            })
        
        return {"input": inputs}

    def result_processor(batch_result):
        return batch_result

    return _make_batch_request(
        image_paths, 
        invoke_url, 
        headers, 
        batch_size, 
        payload_processor, 
        result_processor, 
        "table structure"
    )





def extract_ocr_text(image_path, api_key=None):
    """
    Extract text from an image using NVIDIA OCR API.
    
    Args:
        image_path (str): Path to the image file
        api_key (str, optional): API key for authorization. If not provided, 
                                assumes running in NGC environment
    
    Returns:
        dict: JSON response containing OCR text results
    """
    invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    assert len(image_b64) < 180_000, \
      "To upload larger images, use the assets API (see docs)"

    # Set authorization header
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    else:
        headers = {
            "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
            "Accept": "application/json"
        }

    payload = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:image/png;base64,{image_b64}"
            }
        ]
    }

    print(f"Processing individual OCR inference for: {image_path}")
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"OCR API request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()


def extract_ocr_text_batch(image_paths, api_key=None, batch_size=25, parallel=True, max_workers=None):
    """
    Extract text from multiple images using NVIDIA OCR API in batches.
    
    Args:
        image_paths (list): List of paths to image files
        api_key (str, optional): API key for authorization. If not provided, 
                                assumes running in NGC environment
        batch_size (int): Number of images to process in each batch (default: 25)
        parallel (bool): Whether to process batches in parallel (default: True)
        max_workers (int, optional): Maximum number of worker threads for parallel processing
    
    Returns:
        list: List of JSON responses containing OCR text results for each image
    """
    invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-ocr-v1"
    
    # Set authorization header
    # If api_key is None, try to get it from environment variable
    if api_key is None:
        api_key = os.getenv('NVIDIA_API_KEY')
    
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    else:
        headers = {
            "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
            "Accept": "application/json"
        }

    def payload_processor(batch_paths):
        inputs = []
        for img_path in batch_paths:
            with open(img_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            
            assert len(image_b64) < 180_000, \
              f"Image {img_path} is too large. To upload larger images, use the assets API (see docs)"
            
            inputs.append({
                "type": "image_url",
                "url": f"data:image/jpeg;base64,{image_b64}"
            })
        
        return {"input": inputs}

    def result_processor(batch_result):
        return batch_result

    return _make_batch_request(
        image_paths, 
        invoke_url, 
        headers, 
        batch_size, 
        payload_processor, 
        result_processor, 
        "OCR",
        parallel=parallel,
        max_workers=max_workers
    )





def extract_graphic_elements(image_path, api_key=None):
    """
    Extract graphic elements from an image using NVIDIA AI API.
    
    Args:
        image_path (str): Path to the image file
        api_key (str, optional): API key for authorization. If not provided, 
                                assumes running in NGC environment
    
    Returns:
        dict: JSON response containing graphic elements information
    """
    invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1"

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode()

    assert len(image_b64) < 180_000, \
      "To upload larger images, use the assets API (see docs)"

    # Set authorization header
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    else:
        headers = {
            "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
            "Accept": "application/json"
        }

    payload = {
        "input": [
            {
                "type": "image_url",
                "url": f"data:image/png;base64,{image_b64}"
            }
        ]
    }

    print(f"Processing individual graphic elements inference for: {image_path}")
    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Graphic elements API request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()

def extract_graphic_elements_batch(image_paths, api_key=None, batch_size=5):
    """
    Extract graphic elements from multiple images using NVIDIA AI API in batches.
    
    Args:
        image_paths (list): List of paths to image files
        api_key (str, optional): API key for authorization. If not provided, 
                                assumes running in NGC environment
        batch_size (int): Number of images to process in each batch (default: 5)
    
    Returns:
        list: List of JSON responses containing graphic elements information for each image
    """
    invoke_url = "https://ai.api.nvidia.com/v1/cv/nvidia/nemoretriever-graphic-elements-v1"
    
    # Set authorization header
    # If api_key is None, try to get it from environment variable
    if api_key is None:
        api_key = os.getenv('NVIDIA_API_KEY')
    
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        }
    else:
        headers = {
            "Authorization": "Bearer $API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC",
            "Accept": "application/json"
        }

    def payload_processor(batch_paths):
        inputs = []
        for img_path in batch_paths:
            with open(img_path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            
            assert len(image_b64) < 180_000, \
              f"Image {img_path} is too large. To upload larger images, use the assets API (see docs)"
            
            inputs.append({
                "type": "image_url",
                "url": f"data:image/jpeg;base64,{image_b64}"
            })
        
        return {"input": inputs}

    def result_processor(batch_result):
        return batch_result

    return _make_batch_request(
        image_paths, 
        invoke_url, 
        headers, 
        batch_size, 
        payload_processor, 
        result_processor, 
        "graphic elements"
    )

def process_page_images(pages_dir="pages", output_dir="page_elements", timing=False, ocr_titles=True, batch_processing=True, batch_size=25, pdf_extraction_time=0, print_timing_summary=True):
    # Track all timing values to return when needed
    page_elements_time = 0
    table_structure_time = 0
    chart_structure_time = 0
    ocr_time = 0
    """
    Process all page images in the specified directory, extract content elements,
    and save them in subdirectories organized by content type in JSONL format.
    Uses batch processing for page element extraction, table structure extraction,
    and OCR with fallback to single image processing.
    
    Args:
        pages_dir (str): Directory containing page images
        output_dir (str): Output directory for extracted elements
        timing (bool): Whether to track and report timing for each stage
        ocr_titles (bool): Whether to perform OCR on title elements, defaults to True
        batch_processing (bool): Whether to use batch processing for API calls (default: True)
        batch_size (int): Batch size for API calls (default: 25)
        pdf_extraction_time (float): Time taken for PDF extraction (default: 0)
    """
    # Initialize timing variables
    if timing:
        page_elements_time = 0
        table_structure_time = 0
        chart_structure_time = 0
        ocr_time = 0
    
    # Get API key from environment variable
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("NVIDIA_API_KEY environment variable not set. Skipping page element extraction.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to collect tasks for batch processing later
    table_structure_tasks = []
    table_cell_ocr_tasks = []
    chart_element_ocr_tasks = []
    chart_graphic_elements_tasks = []  # New list for chart graphic elements
    title_ocr_tasks = []
    
    # Process all page images
    page_images = glob(os.path.join(pages_dir, "*.jpg")) + glob(os.path.join(pages_dir, "*.png"))
    
    if batch_processing:
        print(f"Processing {len(page_images)} page images using batch processing with batch size {batch_size}...")
        
        # Process page images in batches for page element detection
        for i in range(0, len(page_images), batch_size):
            batch_paths = page_images[i:i + batch_size]
            print(f"Processing page elements batch: {batch_paths}")
            
            if timing:
                start_time = time.time()
            try:
                batch_results = extract_bounding_boxes_batch(batch_paths, api_key, batch_size)
                
                if timing:
                    page_elements_time += time.time() - start_time
                
                # Process the batch results
                # The batch_results is a list where each element corresponds to a batch
                batch_result = batch_results[i // batch_size]  # Get the corresponding batch result
                
                # The API response structure has 'data' field which contains page data for each image in the batch
                if 'data' in batch_result:
                    batch_page_data = batch_result['data']
                    
                    for img_idx, img_path in enumerate(batch_paths):
                        print(f"Processing {img_path}...")
                        if img_idx < len(batch_page_data):
                            page_data = batch_page_data[img_idx]
                        else:
                            print(f"No data found in response for {img_path}")
                            continue
                        

                        
                        if 'bounding_boxes' in page_data:
                            bounding_boxes = page_data['bounding_boxes']
                            
                            # Open the original image to crop sub-images
                            original_image = Image.open(img_path)
                            img_width, img_height = original_image.size
                        
                        # Process each content type (table, chart, title, etc.)
                        for content_type, elements in bounding_boxes.items():
                            if elements:  # If there are elements of this type
                                content_type_dir = os.path.join(output_dir, content_type)
                                os.makedirs(content_type_dir, exist_ok=True)
                                
                                # Create JSONL file for this content type
                                jsonl_filename = os.path.join(content_type_dir, f"{os.path.basename(img_path).split('.')[0]}_elements.jsonl")
                                
                                # Process each element and save sub-image
                                for idx, element in enumerate(elements):
                                    # Add content type to the element for clarity
                                    element_with_type = element.copy()
                                    element_with_type['type'] = content_type
                                    element_with_type['image_path'] = img_path
                                    
                                    # Calculate pixel coordinates from normalized coordinates
                                    x_min = int(element['x_min'] * img_width)
                                    y_min = int(element['y_min'] * img_height)
                                    x_max = int(element['x_max'] * img_width)
                                    y_max = int(element['y_max'] * img_height)
                                    
                                    # Crop the sub-image based on bounding box
                                    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
                                    
                                    # Save the cropped image in the content type directory
                                    image_filename = os.path.join(content_type_dir, f"{os.path.basename(img_path).split('.')[0]}_element_{idx+1}_{content_type}.jpg")
                                    cropped_image.save(image_filename, "JPEG", quality=90)
                                    
                                    # Add the sub-image path to the element data
                                    element_with_type['sub_image_path'] = image_filename
                                    
                                    # If this is a table, track for table structure extraction
                                    if content_type == 'table':
                                        # Save the cropped image temporarily for API call
                                        temp_table_path = image_filename.replace('.jpg', '_for_api.jpg')
                                        cropped_image.save(temp_table_path, "JPEG", quality=80)
                                        
                                        # Add table structure task to be processed in batch later
                                        table_structure_tasks.append({
                                            'temp_path': temp_table_path,
                                            'image_path': image_filename,
                                            'cropped_image': cropped_image,
                                            'element_with_type': element_with_type
                                        })
                                    
                                    # If this is a chart, track for graphic elements extraction (to be processed in batches)
                                    elif content_type == 'chart':
                                        # Save the cropped image temporarily for API call
                                        temp_chart_path = image_filename.replace('.jpg', '_for_api.jpg')
                                        cropped_image.save(temp_chart_path, "JPEG", quality=80)
                                        
                                        # Add chart graphic elements task to be processed in batch later
                                        chart_graphic_elements_tasks.append({
                                            'temp_path': temp_chart_path,
                                            'image_path': image_filename,
                                            'cropped_image': cropped_image,
                                            'element_with_type': element_with_type
                                        })
                                    
                                    # If this is a title, track the image for OCR if requested
                                    elif content_type == 'title':
                                        if ocr_titles:
                                            # Add title image to OCR tasks to process later in batches
                                            title_ocr_tasks.append(image_filename)
                                    
                                    with open(jsonl_filename, 'a') as f:
                                        f.write(json.dumps(element_with_type) + '\n')
                                        
                        # Close the original image to free memory
                        original_image.close()
            except Exception as e:
                print(f"Error processing page elements batch: {str(e)}")
                # If batch processing fails, fall back to individual processing
                for img_path in batch_paths:
                    print(f"Falling back to processing {img_path} individually...")
                    # We would need to call single image processing here
                    pass
    else:
        # Process page images individually if batch processing is disabled
        for image_path in sorted(page_images):
            print(f"Processing {image_path}...")
            
            try:
                # Extract bounding boxes (page elements)
                if timing:
                    start_time = time.time()
                result = extract_bounding_boxes(image_path, api_key)
                if timing:
                    page_elements_time += time.time() - start_time
                

                
                # Process the bounding box data according to the actual API response format
                if 'data' in result and result['data']:
                    for page_data in result['data']:  # Each page's data
                        if 'bounding_boxes' in page_data:
                            bounding_boxes = page_data['bounding_boxes']
                            
                            # Open the original image to crop sub-images
                            original_image = Image.open(image_path)
                            img_width, img_height = original_image.size
                            
                            # Process each content type (table, chart, title, etc.)
                            for content_type, elements in bounding_boxes.items():
                                if elements:  # If there are elements of this type
                                    content_type_dir = os.path.join(output_dir, content_type)
                                    os.makedirs(content_type_dir, exist_ok=True)
                                    
                                    # Create JSONL file for this content type
                                    jsonl_filename = os.path.join(content_type_dir, f"{os.path.basename(image_path).split('.')[0]}_elements.jsonl")
                                    
                                    # Process each element and save sub-image
                                    for idx, element in enumerate(elements):
                                        # Add content type to the element for clarity
                                        element_with_type = element.copy()
                                        element_with_type['type'] = content_type
                                        element_with_type['image_path'] = image_path
                                        
                                        # Calculate pixel coordinates from normalized coordinates
                                        x_min = int(element['x_min'] * img_width)
                                        y_min = int(element['y_min'] * img_height)
                                        x_max = int(element['x_max'] * img_width)
                                        y_max = int(element['y_max'] * img_height)
                                        
                                        # Crop the sub-image based on bounding box
                                        cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
                                        
                                        # Save the cropped image in the content type directory
                                        image_filename = os.path.join(content_type_dir, f"{os.path.basename(image_path).split('.')[0]}_element_{idx+1}_{content_type}.jpg")
                                        cropped_image.save(image_filename, "JPEG", quality=90)
                                        
                                        # Add the sub-image path to the element data
                                        element_with_type['sub_image_path'] = image_filename
                                        
                                        # If this is a table, track for table structure extraction
                                        if content_type == 'table':
                                            # Save the cropped image temporarily for API call
                                            temp_table_path = image_filename.replace('.jpg', '_for_api.jpg')
                                            cropped_image.save(temp_table_path, "JPEG", quality=80)
                                            
                                            # Add table structure task to be processed in batch later
                                            table_structure_tasks.append({
                                                'temp_path': temp_table_path,
                                                'image_path': image_filename,
                                                'cropped_image': cropped_image,
                                                'element_with_type': element_with_type
                                            })
                                        
                                        # If this is a chart, extract graphic elements
                                        elif content_type == 'chart':
                                            # Save the cropped image temporarily for API call
                                            temp_chart_path = image_filename.replace('.jpg', '_for_api.jpg')
                                            cropped_image.save(temp_chart_path, "JPEG", quality=80)
                                            
                                            # Extract graphic elements from the chart
                                            try:
                                                if timing:
                                                    start_time = time.time()
                                                graphic_elements = extract_graphic_elements(temp_chart_path, api_key)
                                                if timing:
                                                    chart_structure_time += time.time() - start_time
                                                
                                                # Save the graphic elements as a JSON file
                                                elements_filename = image_filename.replace('.jpg', '_elements.json')
                                                with open(elements_filename, 'w') as f:
                                                    json.dump(graphic_elements, f, indent=2)
                                                
                                                # Add elements file path to the element data
                                                element_with_type['elements_path'] = elements_filename
                                                
                                                # Create a subdirectory for chart elements
                                                chart_elements_dir = image_filename.replace('.jpg', '_elements')
                                                os.makedirs(chart_elements_dir, exist_ok=True)
                                                
                                                # Extract element images from graphic elements
                                                if 'data' in graphic_elements and graphic_elements['data']:
                                                    for page_elements in graphic_elements['data']:
                                                        if 'bounding_boxes' in page_elements:
                                                            # Process all types of graphic elements (labels, axes, etc.)
                                                            for elem_type, elem_list in page_elements['bounding_boxes'].items():
                                                                if elem_list and isinstance(elem_list, list):
                                                                    # Get the dimensions of the cropped chart image to convert normalized coordinates
                                                                    chart_width, chart_height = cropped_image.size
                                                                    
                                                                    for elem_idx, elem in enumerate(elem_list):
                                                                        if 'x_min' in elem and 'y_min' in elem and 'x_max' in elem and 'y_max' in elem:
                                                                            # Calculate pixel coordinates from normalized coordinates
                                                                            elem_x_min = int(elem['x_min'] * chart_width)
                                                                            elem_y_min = int(elem['y_min'] * chart_height)
                                                                            elem_x_max = int(elem['x_max'] * chart_width)
                                                                            elem_y_max = int(elem['y_max'] * chart_height)
                                                                        
                                                                            # Ensure coordinates are within image bounds
                                                                            elem_x_min = max(0, elem_x_min)
                                                                            elem_y_min = max(0, elem_y_min)
                                                                            elem_x_max = min(chart_width, elem_x_max)
                                                                            elem_y_max = min(chart_height, elem_y_max)
                                                                        
                                                                            # Crop the element from the chart image
                                                                            if elem_x_max > elem_x_min and elem_y_max > elem_y_min:
                                                                                elem_image = cropped_image.crop((elem_x_min, elem_y_min, elem_x_max, elem_y_max))
                                                                            
                                                                                # Create filename for the element image
                                                                                base_name = os.path.basename(image_filename).replace('.jpg', '')
                                                                                elem_image_filename = os.path.join(chart_elements_dir, f"{base_name}_{elem_type}_{elem_idx+1}.jpg")
                                                                                elem_image.save(elem_image_filename, "JPEG", quality=90)
                                                                            
                                                                                # Add chart element image to OCR tasks to process later in batches
                                                                                chart_element_ocr_tasks.append(elem_image_filename)
                                                
                                                # Remove temporary image used for API call
                                                os.remove(temp_chart_path)
                                            except Exception as e:
                                                print(f"Error extracting graphic elements for {image_filename}: {str(e)}")
                                        
                                        # If this is a title, track the image for OCR if requested
                                        elif content_type == 'title':
                                            if ocr_titles:
                                                # Add title image to OCR tasks to process later in batches
                                                title_ocr_tasks.append(image_filename)
                                        
                                        with open(jsonl_filename, 'a') as f:
                                            f.write(json.dumps(element_with_type) + '\n')
                                            
                            # Close the original image to free memory
                            original_image.close()
                else:
                    print(f"No data found in response for {image_path}")
                    print(f"Full response: {result}")
            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")
    
    # Now process all the table structure tasks in batches
    if table_structure_tasks:
        print(f"Processing {len(table_structure_tasks)} table structure tasks in parallel batches...")
        
        # Extract the temporary image paths to process in batch
        temp_paths = [task['temp_path'] for task in table_structure_tasks]
        
        try:
            if timing:
                start_time = time.time()
            # Process all table structure batches in parallel
            table_structure_batch_results = extract_table_structure_batch(temp_paths, api_key, batch_size)
            # print(f"DEBUG: Table structure batch API returned {len(table_structure_batch_results)} batch result objects")
            # if table_structure_batch_results:
            #     print(f"DEBUG: First table batch result keys: {list(table_structure_batch_results[0].keys()) if isinstance(table_structure_batch_results[0], dict) else type(table_structure_batch_results[0])}")
            if timing:
                table_structure_time += time.time() - start_time
            
            # Process the table structure batch results
            # Flatten the batch results since extract_table_structure_batch now returns all batches together
            all_batch_table_structures = []
            for batch_result in table_structure_batch_results:
                if 'data' in batch_result:
                    all_batch_table_structures.extend(batch_result['data'])
            
            # print(f"DEBUG: Processing {len(table_structure_tasks)} table structure tasks with {len(all_batch_table_structures)} batch results")
            
            # Process each table structure result
            for task_idx, task in enumerate(table_structure_tasks):
                temp_path = task['temp_path']
                image_path = task['image_path']
                cropped_image = task['cropped_image']
                element_with_type = task['element_with_type']
                
                if task_idx < len(all_batch_table_structures):
                    # all_batch_table_structures already contains the structure data directly from the API response
                    table_structure = all_batch_table_structures[task_idx]
                    # print(f"DEBUG: Processing table {task_idx} from batch results. Table structure keys: {list(table_structure.keys()) if isinstance(table_structure, dict) else type(table_structure)}")
                else:
                    # Fallback: process individually if batch results don't match
                    print(f"Batch result mismatch for {temp_path}, processing individually")
                    table_structure = extract_table_structure(temp_path, api_key)
                
                # Save the table structure as a JSON file
                structure_filename = image_path.replace('.jpg', '_structure.json')
                with open(structure_filename, 'w') as f:
                    json.dump(table_structure, f, indent=2)
                
                # Add structure file path to the element data
                element_with_type['structure_path'] = structure_filename
                
                # Create a subdirectory for table cells
                table_cells_dir = image_path.replace('.jpg', '_cells')
                os.makedirs(table_cells_dir, exist_ok=True)
                
                # Extract cell images from table structure
                # After batch processing, table structure has bounding_boxes directly (not under 'data' field)
                initial_table_ocr_count = len(table_cell_ocr_tasks)
                if 'bounding_boxes' in table_structure and 'cell' in table_structure['bounding_boxes']:
                    cells = table_structure['bounding_boxes']['cell']
                    total_cells_found = len(cells)
                    # print(f"DEBUG: Found {total_cells_found} cells in table structure")
                    
                    # Get the dimensions of the cropped table image to convert normalized coordinates
                    table_width, table_height = cropped_image.size
                    
                    for cell_idx, cell in enumerate(cells):
                        # Calculate pixel coordinates from normalized coordinates
                        cell_x_min = int(cell['x_min'] * table_width)
                        cell_y_min = int(cell['y_min'] * table_height)
                        cell_x_max = int(cell['x_max'] * table_width)
                        cell_y_max = int(cell['y_max'] * table_height)
                        
                        # Ensure coordinates are within image bounds
                        cell_x_min = max(0, cell_x_min)
                        cell_y_min = max(0, cell_y_min)
                        cell_x_max = min(table_width, cell_x_max)
                        cell_y_max = min(table_height, cell_y_max)
                        
                        # Crop the cell from the table image
                        if cell_x_max > cell_x_min and cell_y_max > cell_y_min:
                            cell_image = cropped_image.crop((cell_x_min, cell_y_min, cell_x_max, cell_y_max))
                            
                            # Create filename based on source, page number, element number, and cell number
                            base_name = os.path.basename(image_path).replace('.jpg', '')
                            cell_image_filename = os.path.join(table_cells_dir, f"{base_name}_cell_{cell_idx+1}.jpg")
                            cell_image.save(cell_image_filename, "JPEG", quality=90)
                            
                            # Add cell image to OCR tasks to process later in batches
                            table_cell_ocr_tasks.append(cell_image_filename)
                    # print(f"DEBUG: Table {os.path.basename(image_path)} - Found {total_cells_found} total cells in structure")
                # No 'else' needed here - just continue with logging if no cells found
                
                # Log how many cells were added for this table
                added_count = len(table_cell_ocr_tasks) - initial_table_ocr_count
                # print(f"DEBUG: Table {os.path.basename(image_path)} - Actually added {added_count} cells to OCR tasks, total table OCR tasks now: {len(table_cell_ocr_tasks)}")
                
                # Remove temporary image used for API call
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            print(f"Error processing table structure batches in parallel: {str(e)}")
            print("Falling back to sequential processing...")
            # Fall back to sequential processing if parallel processing fails
            for task in table_structure_tasks:
                temp_path = task['temp_path']
                image_path = task['image_path']
                cropped_image = task['cropped_image']
                element_with_type = task['element_with_type']
                
                try:
                    if timing:
                        start_time = time.time()
                    table_structure = extract_table_structure(temp_path, api_key)
                    if timing:
                        table_structure_time += time.time() - start_time
                    
                    # Save the table structure as a JSON file
                    structure_filename = image_path.replace('.jpg', '_structure.json')
                    with open(structure_filename, 'w') as f:
                        json.dump(table_structure, f, indent=2)
                    
                    # Add structure file path to the element data
                    element_with_type['structure_path'] = structure_filename
                    
                    # Create a subdirectory for table cells
                    table_cells_dir = image_path.replace('.jpg', '_cells')
                    os.makedirs(table_cells_dir, exist_ok=True)
                    
                    # Extract cell images from table structure
                    initial_table_ocr_count = len(table_cell_ocr_tasks)
                    # Check both possible structures: original (with 'data' field) and direct structure
                    if 'data' in table_structure and table_structure['data']:
                        # Original structure: results under 'data' field (for fallback individual processing)
                        for page_struct in table_structure['data']:
                            if 'bounding_boxes' in page_struct and 'cell' in page_struct['bounding_boxes']:
                                cells = page_struct['bounding_boxes']['cell']
                                
                                # Get the dimensions of the cropped table image to convert normalized coordinates
                                table_width, table_height = cropped_image.size
                                
                                for cell_idx, cell in enumerate(cells):
                                    # Calculate pixel coordinates from normalized coordinates
                                    cell_x_min = int(cell['x_min'] * table_width)
                                    cell_y_min = int(cell['y_min'] * table_height)
                                    cell_x_max = int(cell['x_max'] * table_width)
                                    cell_y_max = int(cell['y_max'] * table_height)
                                    
                                    # Ensure coordinates are within image bounds
                                    cell_x_min = max(0, cell_x_min)
                                    cell_y_min = max(0, cell_y_min)
                                    cell_x_max = min(table_width, cell_x_max)
                                    cell_y_max = min(table_height, cell_y_max)
                                    
                                    # Crop the cell from the table image
                                    if cell_x_max > cell_x_min and cell_y_max > cell_y_min:
                                        cell_image = cropped_image.crop((cell_x_min, cell_y_min, cell_x_max, cell_y_max))
                                        
                                        # Create filename based on source, page number, element number, and cell number
                                        base_name = os.path.basename(image_path).replace('.jpg', '')
                                        cell_image_filename = os.path.join(table_cells_dir, f"{base_name}_cell_{cell_idx+1}.jpg")
                                        cell_image.save(cell_image_filename, "JPEG", quality=90)
                                        
                                        # Add cell image to OCR tasks to process later in batches
                                        table_cell_ocr_tasks.append(cell_image_filename)
                    elif 'bounding_boxes' in table_structure and 'cell' in table_structure['bounding_boxes']:
                        # Direct structure (for fallback individual processing with same structure as batch)
                        cells = table_structure['bounding_boxes']['cell']
                        
                        # Get the dimensions of the cropped table image to convert normalized coordinates
                        table_width, table_height = cropped_image.size
                        
                        for cell_idx, cell in enumerate(cells):
                            # Calculate pixel coordinates from normalized coordinates
                            cell_x_min = int(cell['x_min'] * table_width)
                            cell_y_min = int(cell['y_min'] * table_height)
                            cell_x_max = int(cell['x_max'] * table_width)
                            cell_y_max = int(cell['y_max'] * table_height)
                            
                            # Ensure coordinates are within image bounds
                            cell_x_min = max(0, cell_x_min)
                            cell_y_min = max(0, cell_y_min)
                            cell_x_max = min(table_width, cell_x_max)
                            cell_y_max = min(table_height, cell_y_max)
                            
                            # Crop the cell from the table image
                            if cell_x_max > cell_x_min and cell_y_max > cell_y_min:
                                cell_image = cropped_image.crop((cell_x_min, cell_y_min, cell_x_max, cell_y_max))
                                
                                # Create filename based on source, page number, element number, and cell number
                                base_name = os.path.basename(image_path).replace('.jpg', '')
                                cell_image_filename = os.path.join(table_cells_dir, f"{base_name}_cell_{cell_idx+1}.jpg")
                                cell_image.save(cell_image_filename, "JPEG", quality=90)
                                
                                # Add cell image to OCR tasks to process later in batches
                                table_cell_ocr_tasks.append(cell_image_filename)
                    
                    # Log how many cells were added for this table
                    added_count = len(table_cell_ocr_tasks) - initial_table_ocr_count
                    # print(f"DEBUG: Table {os.path.basename(image_path)} - Added {added_count} cells to OCR tasks, total table OCR tasks now: {len(table_cell_ocr_tasks)}")
                    
                    # Remove temporary image used for API call
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e_individual:
                    print(f"Error processing table structure for {temp_path}: {str(e_individual)}")

    # Now process all the chart graphic elements tasks in batches
    if chart_graphic_elements_tasks:
        print(f"Processing {len(chart_graphic_elements_tasks)} chart graphic elements tasks in parallel batches...")
        
        # Extract the temporary image paths to process in batch
        temp_paths = [task['temp_path'] for task in chart_graphic_elements_tasks]
        
        try:
            if timing:
                start_time = time.time()
            # Process all chart graphic elements batches in parallel
            chart_graphic_elements_batch_results = extract_graphic_elements_batch(temp_paths, api_key, batch_size)
            # print(f"DEBUG: Chart graphic elements batch API returned {len(chart_graphic_elements_batch_results)} batch result objects")
            # if chart_graphic_elements_batch_results:
            #     print(f"DEBUG: First batch result keys: {list(chart_graphic_elements_batch_results[0].keys()) if isinstance(chart_graphic_elements_batch_results[0], dict) else type(chart_graphic_elements_batch_results[0])}")
            if timing:
                chart_structure_time += time.time() - start_time
            
            # Process the chart graphic elements batch results
            # chart_graphic_elements_batch_results is a list of batch responses
            # Each batch response contains 'data' field with results for all images in that batch
            all_batch_chart_elements = []
            for batch_response in chart_graphic_elements_batch_results:
                # Each batch_response is the API response for one batch call (containing multiple images)
                # It should have a 'data' field containing a list of individual results for each image in the batch
                if isinstance(batch_response, dict) and 'data' in batch_response and isinstance(batch_response['data'], list):
                    # Add each individual result from this batch response 
                    all_batch_chart_elements.extend(batch_response['data'])
                else:
                    # If the response doesn't have expected structure, add as single result
                    all_batch_chart_elements.append(batch_response)
            
            # Process each chart graphic elements result
            # print(f"DEBUG: Processing {len(chart_graphic_elements_tasks)} chart graphic elements tasks with {len(all_batch_chart_elements)} batch results")
            for task_idx, task in enumerate(chart_graphic_elements_tasks):
                temp_path = task['temp_path']
                image_path = task['image_path']
                cropped_image = task['cropped_image']
                element_with_type = task['element_with_type']
                
                if task_idx < len(all_batch_chart_elements):
                    # all_batch_chart_elements contains the full graphic elements response for each image
                    chart_elements = all_batch_chart_elements[task_idx]
                    # print(f"DEBUG: Processing chart {task_idx} from batch results. Chart elements keys: {list(chart_elements.keys()) if isinstance(chart_elements, dict) else type(chart_elements)}")
                else:
                    # Fallback: process individually if batch results don't match
                    print(f"Batch result mismatch for {temp_path}, processing individually")
                    chart_elements = extract_graphic_elements(temp_path, api_key)  # This is the full result
                
                # Save the graphic elements as a JSON file
                elements_filename = image_path.replace('.jpg', '_elements.json')
                with open(elements_filename, 'w') as f:
                    json.dump(chart_elements, f, indent=2)
                
                # Add elements file path to the element data
                element_with_type['elements_path'] = elements_filename
                
                # Create a subdirectory for chart elements
                chart_elements_dir = image_path.replace('.jpg', '_elements')
                os.makedirs(chart_elements_dir, exist_ok=True)
                
                # Extract element images from graphic elements
                # The chart_elements response has bounding_boxes directly (not under 'data' field)
                if 'bounding_boxes' in chart_elements:
                    # Process all types of graphic elements (labels, axes, etc.)
                    for elem_type, elem_list in chart_elements['bounding_boxes'].items():
                                if elem_list and isinstance(elem_list, list):
                                    # Get the dimensions of the cropped chart image to convert normalized coordinates
                                    chart_width, chart_height = cropped_image.size
                                    
                                    for elem_idx, elem in enumerate(elem_list):
                                        if 'x_min' in elem and 'y_min' in elem and 'x_max' in elem and 'y_max' in elem:
                                            # Calculate pixel coordinates from normalized coordinates
                                            elem_x_min = int(elem['x_min'] * chart_width)
                                            elem_y_min = int(elem['y_min'] * chart_height)
                                            elem_x_max = int(elem['x_max'] * chart_width)
                                            elem_y_max = int(elem['y_max'] * chart_height)
                                            
                                            # Ensure coordinates are within image bounds
                                            elem_x_min = max(0, elem_x_min)
                                            elem_y_min = max(0, elem_y_min)
                                            elem_x_max = min(chart_width, elem_x_max)
                                            elem_y_max = min(chart_height, elem_y_max)
                                            
                                            # Crop the element from the chart image
                                            if elem_x_max > elem_x_min and elem_y_max > elem_y_min:
                                                elem_image = cropped_image.crop((elem_x_min, elem_y_min, elem_x_max, elem_y_max))
                                                
                                                # Create filename for the element image
                                                base_name = os.path.basename(image_path).replace('.jpg', '')
                                                elem_image_filename = os.path.join(chart_elements_dir, f"{base_name}_{elem_type}_{elem_idx+1}.jpg")
                                                elem_image.save(elem_image_filename, "JPEG", quality=90)
                                                
                                                # Add chart element image to OCR tasks to process later in batches
                                                chart_element_ocr_tasks.append(elem_image_filename)
                
                # Remove temporary image used for API call
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except Exception as e:
            print(f"Error processing chart graphic elements batches in parallel: {str(e)}")
            print("Falling back to sequential processing...")
            # Fall back to sequential processing if parallel processing fails
            for task in chart_graphic_elements_tasks:
                temp_path = task['temp_path']
                image_path = task['image_path']
                cropped_image = task['cropped_image']
                element_with_type = task['element_with_type']
                
                try:
                    if timing:
                        start_time = time.time()
                    chart_elements = extract_graphic_elements(temp_path, api_key)
                    if timing:
                        chart_structure_time += time.time() - start_time
                    
                    # Save the graphic elements as a JSON file
                    elements_filename = image_path.replace('.jpg', '_elements.json')
                    with open(elements_filename, 'w') as f:
                        json.dump(chart_elements, f, indent=2)
                    
                    # Add elements file path to the element data
                    element_with_type['elements_path'] = elements_filename
                    
                    # Create a subdirectory for chart elements
                    chart_elements_dir = image_path.replace('.jpg', '_elements')
                    os.makedirs(chart_elements_dir, exist_ok=True)
                    
                    # Extract element images from graphic elements
                    initial_chart_ocr_count = len(chart_element_ocr_tasks)
                    # Sequential processing: check original structure first, then fallback structure
                    if 'data' in chart_elements and chart_elements['data']:
                        # Original structure: results under 'data' field
                        for page_elements in chart_elements['data']:
                            if 'bounding_boxes' in page_elements:
                                # Process all types of graphic elements (labels, axes, etc.)
                                for elem_type, elem_list in page_elements['bounding_boxes'].items():
                                    if elem_list and isinstance(elem_list, list):
                                        # Get the dimensions of the cropped chart image to convert normalized coordinates
                                        chart_width, chart_height = cropped_image.size
                                        
                                        for elem_idx, elem in enumerate(elem_list):
                                            if 'x_min' in elem and 'y_min' in elem and 'x_max' in elem and 'y_max' in elem:
                                                # Calculate pixel coordinates from normalized coordinates
                                                elem_x_min = int(elem['x_min'] * chart_width)
                                                elem_y_min = int(elem['y_min'] * chart_height)
                                                elem_x_max = int(elem['x_max'] * chart_width)
                                                elem_y_max = int(elem['y_max'] * chart_height)
                                                
                                                # Ensure coordinates are within image bounds
                                                elem_x_min = max(0, elem_x_min)
                                                elem_y_min = max(0, elem_y_min)
                                                elem_x_max = min(chart_width, elem_x_max)
                                                elem_y_max = min(chart_height, elem_y_max)
                                                
                                                # Crop the element from the chart image
                                                if elem_x_max > elem_x_min and elem_y_max > elem_y_min:
                                                    elem_image = cropped_image.crop((elem_x_min, elem_y_min, elem_x_max, elem_y_max))
                                                    
                                                    # Create filename for the element image
                                                    base_name = os.path.basename(image_path).replace('.jpg', '')
                                                    elem_image_filename = os.path.join(chart_elements_dir, f"{base_name}_{elem_type}_{elem_idx+1}.jpg")
                                                    elem_image.save(elem_image_filename, "JPEG", quality=90)
                                                    
                                                    # Add chart element image to OCR tasks to process later in batches
                                                    chart_element_ocr_tasks.append(elem_image_filename)
                    elif 'bounding_boxes' in chart_elements:
                        # Direct structure (for fallback individual processing)
                        # Process all types of graphic elements (labels, axes, etc.)
                        for elem_type, elem_list in chart_elements['bounding_boxes'].items():
                            if elem_list and isinstance(elem_list, list):
                                # Get the dimensions of the cropped chart image to convert normalized coordinates
                                chart_width, chart_height = cropped_image.size
                                
                                for elem_idx, elem in enumerate(elem_list):
                                    if 'x_min' in elem and 'y_min' in elem and 'x_max' in elem and 'y_max' in elem:
                                        # Calculate pixel coordinates from normalized coordinates
                                        elem_x_min = int(elem['x_min'] * chart_width)
                                        elem_y_min = int(elem['y_min'] * chart_height)
                                        elem_x_max = int(elem['x_max'] * chart_width)
                                        elem_y_max = int(elem['y_max'] * chart_height)
                                        
                                        # Ensure coordinates are within image bounds
                                        elem_x_min = max(0, elem_x_min)
                                        elem_y_min = max(0, elem_y_min)
                                        elem_x_max = min(chart_width, elem_x_max)
                                        elem_y_max = min(chart_height, elem_y_max)
                                        
                                        # Crop the element from the chart image
                                        if elem_x_max > elem_x_min and elem_y_max > elem_y_min:
                                            elem_image = cropped_image.crop((elem_x_min, elem_y_min, elem_x_max, elem_y_max))
                                            
                                            # Create filename for the element image
                                            base_name = os.path.basename(image_path).replace('.jpg', '')
                                            elem_image_filename = os.path.join(chart_elements_dir, f"{base_name}_{elem_type}_{elem_idx+1}.jpg")
                                            elem_image.save(elem_image_filename, "JPEG", quality=90)
                                            
                                            # Add chart element image to OCR tasks to process later in batches
                                            chart_element_ocr_tasks.append(elem_image_filename)
                    
                    # Log how many elements were added for this chart
                    added_count = len(chart_element_ocr_tasks) - initial_chart_ocr_count
                    # print(f"DEBUG: Chart {os.path.basename(image_path)} - Added {added_count} elements to OCR tasks, total chart OCR tasks now: {len(chart_element_ocr_tasks)}")
                    
                    # Remove temporary image used for API call
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e_individual:
                    print(f"Error processing graphic elements for {temp_path}: {str(e_individual)}")
    
    # Count OCR tasks by type
    num_table_cell_ocr_tasks = len(table_cell_ocr_tasks)
    num_chart_element_ocr_tasks = len(chart_element_ocr_tasks)
    num_title_ocr_tasks = len(title_ocr_tasks) if ocr_titles else 0
    
    # Now process all the OCR tasks in batches
    # print(f"DEBUG: Preparing OCR tasks - Table cell tasks: {num_table_cell_ocr_tasks}, Chart element tasks: {num_chart_element_ocr_tasks}, Title tasks: {num_title_ocr_tasks}")
    all_ocr_tasks = []
    all_ocr_tasks.extend(table_cell_ocr_tasks)
    all_ocr_tasks.extend(chart_element_ocr_tasks)
    if ocr_titles:
        all_ocr_tasks.extend(title_ocr_tasks)

    # Process OCR tasks in batches
    if all_ocr_tasks:
        print(f"Processing {len(all_ocr_tasks)} OCR tasks in parallel batches...")
        try:
            if timing:
                start_time = time.time()
            # Process all OCR batches in parallel
            ocr_batch_results = extract_ocr_text_batch(all_ocr_tasks, api_key, batch_size, parallel=True)
            if timing:
                ocr_time += time.time() - start_time
            
            # Process the OCR batch results
            # OCR batch processing returns results for all images in the batch
            batch_results_processed = 0
            for batch_result in ocr_batch_results:
                if 'data' in batch_result and isinstance(batch_result['data'], list):
                    # Process each result in this batch (each corresponds to an image in the same order)
                    for result_idx, single_result in enumerate(batch_result['data']):
                        # Calculate the global index in all_ocr_tasks
                        global_idx = batch_results_processed
                        if global_idx < len(all_ocr_tasks):
                            img_path = all_ocr_tasks[global_idx]
                            
                            # Create individual result for this specific image  
                            # The batch result for a single image should have the same structure as individual API call
                            individual_ocr_result = {"data": [single_result]}
                            
                            # Save the OCR result as a JSON file
                            ocr_filename = img_path.replace('.jpg', '_ocr.json')
                            with open(ocr_filename, 'w') as f:
                                json.dump(individual_ocr_result, f, indent=2)
                            
                            batch_results_processed += 1
                        else:
                            print(f"Unexpected: More results in batch than tasks. Skipping result {result_idx}")
                            break
                else:
                    print(f"OCR batch result missing 'data' field or not a list: {batch_result}")
            
            # Check if any tasks were not processed due to mismatched results
            if batch_results_processed < len(all_ocr_tasks):
                print(f"OCR batch processing mismatch: processed {batch_results_processed} of {len(all_ocr_tasks)} tasks. Fallback to individual processing for remaining...")
                # Process remaining items individually
                for remaining_idx in range(batch_results_processed, len(all_ocr_tasks)):
                    img_path = all_ocr_tasks[remaining_idx]
                    try:
                        ocr_result = extract_ocr_text(img_path, api_key)
                        ocr_filename = img_path.replace('.jpg', '_ocr.json')
                        with open(ocr_filename, 'w') as f:
                            json.dump(ocr_result, f, indent=2)
                    except Exception as e:
                        print(f"Error processing OCR for {img_path} individually: {str(e)}")
        except Exception as e:
            print(f"Error processing OCR batches in parallel: {str(e)}")
            print("Falling back to sequential processing...")
            # Fall back to sequential processing if parallel processing fails
            for img_path in all_ocr_tasks:
                try:
                    if timing:
                        start_time = time.time()
                    ocr_result = extract_ocr_text(img_path, api_key)
                    if timing:
                        ocr_time += time.time() - start_time
                    
                    # Save the OCR result as a JSON file
                    ocr_filename = img_path.replace('.jpg', '_ocr.json')
                    with open(ocr_filename, 'w') as f:
                        json.dump(ocr_result, f, indent=2)
                except Exception as e_individual:
                    print(f"Error performing OCR on {img_path}: {str(e_individual)}")

    # Report timing if requested
    if timing and print_timing_summary:
        total_time = pdf_extraction_time + page_elements_time + table_structure_time + chart_structure_time + ocr_time
        print(f"Timing Summary:")
        print(f"  PDF Extraction: {pdf_extraction_time:.2f}s")
        print(f"  Page Elements Inference: {page_elements_time:.2f}s")
        print(f"  Table Structure: {table_structure_time:.2f}s")
        print(f"  Chart Structure: {chart_structure_time:.2f}s")
        print(f"  OCR: {ocr_time:.2f}s")
        print(f"  Total: {total_time:.2f}s")

    # Return timing data if requested (even if not printed)
    if timing:
        return {
            'pdf_extraction_time': pdf_extraction_time,
            'page_elements_time': page_elements_time,
            'table_structure_time': table_structure_time,
            'chart_structure_time': chart_structure_time,
            'ocr_time': ocr_time,
            'ai_processing_time': page_elements_time + table_structure_time + chart_structure_time + ocr_time,
            'ocr_task_counts': {
                'table_cells': num_table_cell_ocr_tasks,
                'chart_elements': num_chart_element_ocr_tasks,
                'titles': num_title_ocr_tasks
            }
        }
    return None


def get_content_counts_with_text_stats(output_dir="page_elements"):
    """
    Generate page-level counts of content types and inference requests made for each sub-page element,
    including text statistics (words, characters, lines).
    
    Args:
        output_dir (str): Output directory where extracted elements are stored
        
    Returns:
        dict: A dictionary containing content type counts, inference request counts, and text statistics
    """
    counts = {
        'pages': {},
        'total_elements': 0,
        'total_inference_requests': 0,
        'content_type_breakdown': {},
        'total_text_stats': {
            'words': 0,
            'chars': 0,
            'lines': 0
        }
    }
    
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return counts
    
    page_elements_dirs = os.listdir(output_dir)
    
    for content_type in page_elements_dirs:
        content_type_path = os.path.join(output_dir, content_type)
        
        if not os.path.isdir(content_type_path):
            continue
            
        counts['content_type_breakdown'][content_type] = {
            'total_elements': 0,
            'inference_requests': 0,
            'text_stats': {
                'words': 0,
                'chars': 0,
                'lines': 0
            }
        }
        
        # Process each JSONL file for this content type
        for jsonl_file in glob(os.path.join(content_type_path, "*_elements.jsonl")):
            page_name = os.path.basename(jsonl_file).replace("_elements.jsonl", "")
            
            if page_name not in counts['pages']:
                counts['pages'][page_name] = {
                    'content_types': {},
                    'inference_requests': 0,
                    'text_stats': {
                        'words': 0,
                        'chars': 0,
                        'lines': 0
                    }
                }
            
            # Count elements in this JSONL file and collect text stats
            element_count = 0
            page_text_stats = {
                'words': 0,
                'chars': 0,
                'lines': 0
            }
            
            with open(jsonl_file, 'r') as f:
                for line in f:
                    if line.strip():
                        element_count += 1
                        data = json.loads(line)
                        
                        # Count inference requests based on the content type
                        page_inference_requests = 0
                        
                        if content_type == 'table':
                            # 1 for table structure + 1 per cell for OCR
                            counts['total_inference_requests'] += 1
                            counts['content_type_breakdown'][content_type]['inference_requests'] += 1
                            page_inference_requests += 1
                            
                            # Count cells if structure file exists
                            if 'structure_path' in data and os.path.exists(data['structure_path']):
                                with open(data['structure_path'], 'r') as struct_file:
                                    struct_data = json.load(struct_file)
                                    cell_count = 0
                                    if 'data' in struct_data and struct_data['data']:
                                        for page_struct in struct_data['data']:
                                            if 'bounding_boxes' in page_struct and 'cell' in page_struct['bounding_boxes']:
                                                cell_count = len(page_struct['bounding_boxes']['cell'])
                                    counts['total_inference_requests'] += cell_count
                                    counts['content_type_breakdown'][content_type]['inference_requests'] += cell_count
                                    page_inference_requests += cell_count
                                    
                                    # Count OCR requests for each cell and collect text stats
                                    cells_dir = data['sub_image_path'].replace('.jpg', '_cells')
                                    if os.path.exists(cells_dir):
                                        cell_files = glob(os.path.join(cells_dir, "*_ocr.json"))
                                        counts['total_inference_requests'] += len(cell_files)
                                        counts['content_type_breakdown'][content_type]['inference_requests'] += len(cell_files)
                                        page_inference_requests += len(cell_files)
                                        
                                        # Collect text stats from each cell's OCR
                                        for cell_file in cell_files:
                                            with open(cell_file, 'r') as ocr_f:
                                                ocr_data = json.load(ocr_f)
                                                if 'data' in ocr_data and ocr_data['data']:
                                                    for ocr_item in ocr_data['data']:
                                                        if 'text_detections' in ocr_item:
                                                            for text_det in ocr_item['text_detections']:
                                                                text = text_det['text_prediction']['text']
                                                                words = len(text.split())
                                                                chars = len(text)
                                                                lines = text.count('\n') + 1
                                                                
                                                                page_text_stats['words'] += words
                                                                page_text_stats['chars'] += chars
                                                                page_text_stats['lines'] += lines
                                                                
                                                                counts['total_text_stats']['words'] += words
                                                                counts['total_text_stats']['chars'] += chars
                                                                counts['total_text_stats']['lines'] += lines
                                                                
                                                                counts['content_type_breakdown'][content_type]['text_stats']['words'] += words
                                                                counts['content_type_breakdown'][content_type]['text_stats']['chars'] += chars
                                                                counts['content_type_breakdown'][content_type]['text_stats']['lines'] += lines
                        
                        elif content_type == 'chart':
                                # 1 for graphic elements + 1 per element for OCR
                                counts['total_inference_requests'] += 1
                                counts['content_type_breakdown'][content_type]['inference_requests'] += 1
                                page_inference_requests += 1
                            
                                # Count OCR requests for chart elements and collect text stats
                                elements_dir = data['sub_image_path'].replace('.jpg', '_elements')
                                if os.path.exists(elements_dir):
                                    ocr_files = glob(os.path.join(elements_dir, "*_ocr.json"))
                                counts['total_inference_requests'] += len(ocr_files)
                                counts['content_type_breakdown'][content_type]['inference_requests'] += len(ocr_files)
                                page_inference_requests += len(ocr_files)
                                
                                # Collect text stats from chart elements' OCR
                                for ocr_file in ocr_files:
                                    with open(ocr_file, 'r') as ocr_f:
                                        ocr_data = json.load(ocr_f)
                                        if 'data' in ocr_data and ocr_data['data']:
                                            for ocr_item in ocr_data['data']:
                                                if 'text_detections' in ocr_item:
                                                    for text_det in ocr_item['text_detections']:
                                                        text = text_det['text_prediction']['text']
                                                        words = len(text.split())
                                                        chars = len(text)
                                                        lines = text.count('\n') + 1
                                                        
                                                        page_text_stats['words'] += words
                                                        page_text_stats['chars'] += chars
                                                        page_text_stats['lines'] += lines
                                                        
                                                        counts['total_text_stats']['words'] += words
                                                        counts['total_text_stats']['chars'] += chars
                                                        counts['total_text_stats']['lines'] += lines
                                                        
                                                        counts['content_type_breakdown'][content_type]['text_stats']['words'] += words
                                                        counts['content_type_breakdown'][content_type]['text_stats']['chars'] += chars
                                                        counts['content_type_breakdown'][content_type]['text_stats']['lines'] += lines
                        
                        else:
                            # For other content types like titles, just OCR on the element
                            counts['total_inference_requests'] += 1
                            counts['content_type_breakdown'][content_type]['inference_requests'] += 1
                            page_inference_requests += 1
                            
                            # Collect text stats from title OCR if available
                            ocr_path = data['sub_image_path'].replace('.jpg', '_ocr.json') if 'sub_image_path' in data else None
                            if ocr_path and os.path.exists(ocr_path):
                                with open(ocr_path, 'r') as ocr_f:
                                    ocr_data = json.load(ocr_f)
                                    if 'data' in ocr_data and ocr_data['data']:
                                        for ocr_item in ocr_data['data']:
                                            if 'text_detections' in ocr_item:
                                                for text_det in ocr_item['text_detections']:
                                                    text = text_det['text_prediction']['text']
                                                    words = len(text.split())
                                                    chars = len(text)
                                                    lines = text.count('\n') + 1
                                                    
                                                    page_text_stats['words'] += words
                                                    page_text_stats['chars'] += chars
                                                    page_text_stats['lines'] += lines
                                                    
                                                    counts['total_text_stats']['words'] += words
                                                    counts['total_text_stats']['chars'] += chars
                                                    counts['total_text_stats']['lines'] += lines
                                                    
                                                    counts['content_type_breakdown'][content_type]['text_stats']['words'] += words
                                                    counts['content_type_breakdown'][content_type]['text_stats']['chars'] += chars
                                                    counts['content_type_breakdown'][content_type]['text_stats']['lines'] += lines
                        
                        # Add to page's inference request count
                        counts['pages'][page_name]['inference_requests'] += page_inference_requests
                        
                        # Add to page's text stats
                        counts['pages'][page_name]['text_stats']['words'] += page_text_stats['words']
                        counts['pages'][page_name]['text_stats']['chars'] += page_text_stats['chars']
                        counts['pages'][page_name]['text_stats']['lines'] += page_text_stats['lines']
            
            counts['total_elements'] += element_count
            counts['content_type_breakdown'][content_type]['total_elements'] = (
                counts['content_type_breakdown'][content_type]['total_elements'] + element_count
            )
            
            if content_type not in counts['pages'][page_name]['content_types']:
                counts['pages'][page_name]['content_types'][content_type] = 0
            counts['pages'][page_name]['content_types'][content_type] += element_count
    
    return counts


def get_content_counts(output_dir="page_elements"):
    """
    Generate page-level counts of content types and inference requests made for each sub-page element.
    This is the original function provided for backward compatibility.
    
    Args:
        output_dir (str): Output directory where extracted elements are stored
        
    Returns:
        dict: A dictionary containing content type counts and inference request counts
    """
    # For backward compatibility, call the enhanced function and return only original fields
    full_counts = get_content_counts_with_text_stats(output_dir)
    
    # Return only the original fields
    simplified_counts = {
        'pages': {},
        'total_elements': full_counts['total_elements'],
        'total_inference_requests': full_counts['total_inference_requests'],
        'content_type_breakdown': {}
    }
    
    for page_name, page_data in full_counts['pages'].items():
        simplified_counts['pages'][page_name] = {
            'content_types': page_data['content_types'],
            'inference_requests': page_data['inference_requests']
        }
    
    for content_type, content_data in full_counts['content_type_breakdown'].items():
        simplified_counts['content_type_breakdown'][content_type] = {
            'total_elements': content_data['total_elements'],
            'inference_requests': content_data['inference_requests']
        }
    
    return simplified_counts


def print_content_summary(output_dir="page_elements"):
    """
    Print a summary of content type counts and inference requests.
    
    Args:
        output_dir (str): Output directory where extracted elements are stored
    """
    counts = get_content_counts(output_dir)
    
    print("Content Type Summary:")
    print("=" * 50)
    print(f"Total Elements: {counts['total_elements']}")
    print()
    
    print("Per-Page Breakdown:")
    for page_name, page_stats in counts['pages'].items():
        content_str = ", ".join([f"{content_type}s: {count}" for content_type, count in page_stats['content_types'].items()])
        print(f"  {page_name}: {content_str}")


def get_all_extracted_content(pages_dir="pages", output_dir="page_elements"):
    """
    Create a single result object exposing all the extracted content texts,
    filepaths to related images on disk, and bounding boxes.
    
    Args:
        pages_dir (str): Directory containing original page images
        output_dir (str): Output directory where extracted elements are stored
        
    Returns:
        dict: A comprehensive result object containing all extracted content
    """
    result = {
        'pages': {},
        'content_elements': {
            'tables': [],
            'charts': [],
            'titles': [],
            'other': []
        },
        'total_elements': 0
    }
    
    # Determine the texts directory based on the pages directory structure
    # In the new extraction structure, pages and texts are both subdirectories of the extraction root
    extraction_root = os.path.dirname(pages_dir)  # This should be the extraction directory like "extracts/document_name"
    texts_dir = os.path.join(extraction_root, "texts")
    
    # Process original page images and text files to add to the result
    for image_path in sorted(glob(os.path.join(pages_dir, "*.jpg")) + glob(os.path.join(pages_dir, "*.png"))):
        page_name = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
        
        # Look for corresponding text file in the texts directory
        text_file_path = os.path.join(texts_dir, f"{page_name}.txt")
        
        page_text = ""
        if os.path.exists(text_file_path):
            with open(text_file_path, 'r', encoding='utf-8') as f:
                page_text = f.read()
        
        if page_name not in result['pages']:
            result['pages'][page_name] = {
                'original_image_path': image_path,
                'page_text': page_text,  # Add the extracted plain text
                'elements': []
            }
    
    # Process each content type directory
    if os.path.exists(output_dir):
        for content_type in os.listdir(output_dir):
            content_type_path = os.path.join(output_dir, content_type)
            
            if not os.path.isdir(content_type_path):
                continue
                
            # Process each JSONL file for this content type
            for jsonl_file in glob(os.path.join(content_type_path, "*_elements.jsonl")):
                page_name = os.path.basename(jsonl_file).replace("_elements.jsonl", "")
                
                if page_name not in result['pages']:
                    result['pages'][page_name] = {
                        'original_image_path': os.path.join(pages_dir, f"{page_name}.jpg"),
                        'elements': []
                    }
                
                # Process each element in the JSONL file
                with open(jsonl_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            element_data = json.loads(line)
                            
                            # Create a comprehensive element record
                            element_record = {
                                'type': element_data.get('type', content_type),
                                'original_image_path': element_data.get('image_path'),
                                'sub_image_path': element_data.get('sub_image_path'),
                                'bounding_box': {
                                    'x_min': element_data.get('x_min'),
                                    'y_min': element_data.get('y_min'),
                                    'x_max': element_data.get('x_max'),
                                    'y_max': element_data.get('y_max'),
                                    'confidence': element_data.get('confidence', 1.0)
                                },
                                'content_texts': [],
                                'related_images': [element_data.get('sub_image_path')] if element_data.get('sub_image_path') else []
                            }
                            
                            # Handle content-specific data
                            if content_type == 'table':
                                # Get table structure and cell data
                                if 'structure_path' in element_data and os.path.exists(element_data['structure_path']):
                                    with open(element_data['structure_path'], 'r') as struct_file:
                                        struct_data = json.load(struct_file)
                                        
                                        # Add table structure info
                                        element_record['table_structure_path'] = element_data['structure_path']
                                        
                                        # Process cells if available
                                        if 'data' in struct_data and struct_data['data']:
                                            for page_struct in struct_data['data']:
                                                if 'bounding_boxes' in page_struct and 'cell' in page_struct['bounding_boxes']:
                                                    cells = page_struct['bounding_boxes']['cell']
                                                    
                                                    # Get cell images and OCR text
                                                    cells_dir = element_data['sub_image_path'].replace('.jpg', '_cells')
                                                    if os.path.exists(cells_dir):
                                                        for cell_idx, cell in enumerate(cells):
                                                            cell_image_path = os.path.join(cells_dir, f"{os.path.basename(element_data['sub_image_path']).replace('.jpg', '')}_cell_{cell_idx+1}.jpg")
                                                            ocr_path = cell_image_path.replace('.jpg', '_ocr.json')
                                                            
                                                            if os.path.exists(cell_image_path):
                                                                element_record['related_images'].append(cell_image_path)
                                                                
                                                                # Extract OCR text if available
                                                                if os.path.exists(ocr_path):
                                                                    with open(ocr_path, 'r') as ocr_file:
                                                                        ocr_data = json.load(ocr_file)
                                                                        if 'data' in ocr_data and ocr_data['data']:
                                                                            for ocr_item in ocr_data['data']:
                                                                                if 'text_detections' in ocr_item:
                                                                                    for text_det in ocr_item['text_detections']:
                                                                                        element_record['content_texts'].append({
                                                                                            'text': text_det['text_prediction']['text'],
                                                                                            'confidence': text_det['text_prediction']['confidence'],
                                                                                            'source': f"cell_{cell_idx+1}",
                                                                                            'bounding_box': text_det.get('bounding_box')
                                                                                        })
                                
                                # Add to tables list
                                result['content_elements']['tables'].append(element_record)
                                
                            elif content_type == 'chart':
                                # Get chart elements and OCR text
                                if 'elements_path' in element_data and os.path.exists(element_data['elements_path']):
                                    element_record['chart_elements_path'] = element_data['elements_path']
                                    
                                    # Get chart element images and OCR
                                    elements_dir = element_data['sub_image_path'].replace('.jpg', '_elements')
                                    if os.path.exists(elements_dir):
                                        # Collect all element images in this directory
                                        for elem_file in glob(os.path.join(elements_dir, "*.jpg")):
                                            element_record['related_images'].append(elem_file)
                                            
                                            # Get OCR text for this element if available
                                            ocr_path = elem_file.replace('.jpg', '_ocr.json')
                                            if os.path.exists(ocr_path):
                                                with open(ocr_path, 'r') as ocr_file:
                                                    ocr_data = json.load(ocr_file)
                                                    if 'data' in ocr_data and ocr_data['data']:
                                                        for ocr_item in ocr_data['data']:
                                                            if 'text_detections' in ocr_item:
                                                                for text_det in ocr_item['text_detections']:
                                                                    element_record['content_texts'].append({
                                                                        'text': text_det['text_prediction']['text'],
                                                                        'confidence': text_det['text_prediction']['confidence'],
                                                                        'source': os.path.basename(elem_file),
                                                                        'bounding_box': text_det.get('bounding_box')
                                                                    })
                                
                                # Add to charts list
                                result['content_elements']['charts'].append(element_record)
                                
                            elif content_type == 'title':
                                # Try to get OCR text for title elements
                                ocr_path = element_data['sub_image_path'].replace('.jpg', '_ocr.json') if 'sub_image_path' in element_data else None
                                if ocr_path and os.path.exists(ocr_path):
                                    with open(ocr_path, 'r') as ocr_file:
                                        ocr_data = json.load(ocr_file)
                                        if 'data' in ocr_data and ocr_data['data']:
                                            for ocr_item in ocr_data['data']:
                                                if 'text_detections' in ocr_item:
                                                    for text_det in ocr_item['text_detections']:
                                                        element_record['content_texts'].append({
                                                            'text': text_det['text_prediction']['text'],
                                                            'confidence': text_det['text_prediction']['confidence'],
                                                            'source': 'title',
                                                            'bounding_box': text_det.get('bounding_box')
                                                        })
                                
                                # Add to titles list
                                result['content_elements']['titles'].append(element_record)
                                
                            else:
                                # Add to other list
                                result['content_elements']['other'].append(element_record)
                            
                            # Add to page elements
                            result['pages'][page_name]['elements'].append(element_record)
                            result['total_elements'] += 1
    
    return result

def format_markdown_table(element, content_texts):
    """
    Format table content as markdown from extracted content texts and structure.
    
    Args:
        element (dict): Element record containing table data
        content_texts (list): List of content text objects from the table element
        
    Returns:
        list: List of markdown lines representing the table
    """
    import os
    import json
    
    markdown_lines = []
    
    # Extract cell texts and organize by source for proper table structure
    cell_texts = []
    other_texts = []
    
    for content in content_texts:
        text = content.get('text', '')
        source = content.get('source', '')
        if source and source.startswith('cell_'):
            # Store cell data with source information
            cell_texts.append((source, text))
        elif text.strip():
            other_texts.append(text)

    # Add any non-cell text first
    for text in other_texts:
        markdown_lines.append(f"> {text.strip()}")

    # Now format cell data as markdown table if there's cell data
    if cell_texts:
        # Sort cells by their source order (cell_1, cell_2, etc.)
        sorted_cells = sorted(cell_texts, key=lambda x: int(x[0].replace('cell_', '')) if x[0].startswith('cell_') else 0)

        # Get the table structure to properly format the table
        table_structure_path = element.get('table_structure_path')
        if table_structure_path and os.path.exists(table_structure_path):
            with open(table_structure_path, 'r') as struct_file:
                struct_data = json.load(struct_file)

            # Extract table structure to determine rows and columns
            if 'data' in struct_data and struct_data['data']:
                for page_struct in struct_data['data']:
                    if 'bounding_boxes' in page_struct and 'cell' in page_struct['bounding_boxes']:
                        cells = page_struct['bounding_boxes']['cell']

                        # Now we'll use both the structure information and the OCR content
                        # to build a proper markdown table

                        # For a proper markdown table, we need to determine the grid structure
                        # from the cell coordinates. Since this is complex, we'll implement
                        # a simplified version that attempts to organize cells into rows.

                        # First, let's get the cell data with coordinates if available
                        # and the actual OCR text content
                        cell_data_with_coords = []
                        for i, cell_info in enumerate(cells):
                            if i < len(sorted_cells):
                                cell_id, text = sorted_cells[i]
                                # Use the structure coordinates and the OCR text
                                cell_data_with_coords.append({
                                    'text': text,
                                    'x_min': cell_info.get('x_min', 0),
                                    'y_min': cell_info.get('y_min', 0),
                                    'x_max': cell_info.get('x_max', 1),
                                    'y_max': cell_info.get('y_max', 1)
                                })

                        # If we have cell data, try to create a proper table structure
                        if cell_data_with_coords:
                            # For simplicity of this implementation, we'll sort cells by y-coordinate first
                            # (for rows) and then by x-coordinate (for columns) to simulate table structure
                            # This is a simplified approach - a robust implementation would use
                            # more sophisticated algorithms to determine table grid

                            # Group cells by similar y-coordinates (rows)
                            # Use a simple approach with tolerance for y-coordinates
                            row_tolerance = 0.05  # Adjust based on coordinate system
                            rows = []
                            used_cells = set()

                            for i, cell in enumerate(cell_data_with_coords):
                                if i in used_cells:
                                    continue

                                current_row = [cell]
                                used_cells.add(i)

                                # Find other cells with similar y_min (in the same row)
                                for j, other_cell in enumerate(cell_data_with_coords):
                                    if j in used_cells:
                                        continue
                                    if abs(cell['y_min'] - other_cell['y_min']) < row_tolerance:
                                        current_row.append(other_cell)
                                        used_cells.add(j)

                                # Sort cells in row by x-coordinate (left to right)
                                current_row.sort(key=lambda c: c['x_min'])
                                rows.append(current_row)

                            # Now sort rows by y-coordinate (top to bottom)
                            rows.sort(key=lambda r: r[0]['y_min'] if r else 0)

                            if rows:
                                # Create markdown table
                                if rows:
                                    # Create header row from first row if it looks like a header
                                    first_row = rows[0]
                                    headers = [cell['text'].strip() if cell['text'].strip() else " " for cell in first_row]
                                    header_row = "| " + " | ".join(headers) + " |"
                                    separator_row = "|" + "|".join([" --- " for _ in headers]) + "|"

                                    markdown_lines.append(header_row)
                                    markdown_lines.append(separator_row)

                                    # Add remaining rows
                                    for row in rows[1:]:
                                        row_data = [cell['text'].strip() if cell['text'].strip() else " " for cell in row]
                                        row_str = "| " + " | ".join(row_data) + " |"
                                        markdown_lines.append(row_str)
                            else:
                                # Fallback: if we cannot determine rows, just put all cells in one row
                                all_texts = [cell['text'].strip() if cell['text'].strip() else " " for cell in cell_data_with_coords]
                                if all_texts:
                                    header_row = "| " + " | ".join(all_texts) + " |"
                                    separator_row = "|" + "|".join([" --- " for _ in all_texts]) + "|"
                                    markdown_lines.append(header_row)
                                    markdown_lines.append(separator_row)
        else:
            # If no structure file, just display as list
            for source, text in sorted_cells:
                if text.strip():
                    markdown_lines.append(f"- {text.strip()}")
    else:
        # If no cell texts, just display other texts (if any)
        for text in other_texts:
            if text.strip():
                markdown_lines.append(f"- {text.strip()}")
    
    return markdown_lines

def format_markdown_chart(element, content_texts):
    """
    Format chart content as markdown from extracted content texts.
    
    Args:
        element (dict): Element record containing chart data
        content_texts (list): List of content text objects from the chart element
        
    Returns:
        list: List of markdown lines representing the chart content
    """
    import os
    
    markdown_lines = []
    
    # Process each text element in the chart
    for content in content_texts:
        text = content.get('text', '')
        source = content.get('source', '')
        
        # Check if this is a chart element with source information
        if source and text.strip():
            # Extract a more meaningful name from the source (remove page info, element info)
            # Example: "page_001_element_1_chart_ylabel_2.jpg" -> "Ylabel 2"
            source_name = source.replace('.jpg', '').replace('.png', '')
            
            # Look for common chart element patterns and extract the meaningful part
            # Handle complex multi-part names first
            if 'chart_title' in source_name:
                clean_source = 'Chart Title'
            elif 'x_label' in source_name or 'xlabel' in source_name:
                clean_source = 'X Label'
            elif 'y_label' in source_name or 'ylabel' in source_name:
                clean_source = 'Y Label'
            elif 'legend' in source_name:
                clean_source = 'Legend'
            elif 'axis' in source_name:
                clean_source = 'Axis'
            else:
                # For other cases, split by "_" and filter out common prefixes
                parts = source_name.split('_')
                
                # Remove common prefixes like 'page', 'element', 'chart', numbers
                meaningful_parts = []
                i = 0
                while i < len(parts):
                    part = parts[i]
                    
                    # Skip common prefixes
                    if part in ['page', 'element', 'chart'] or (part.isdigit() and len(part) <= 3):
                        # Skip this and potentially the next part (if it's a page/element number)
                        i += 1
                        if i < len(parts) and parts[i].isdigit():
                            i += 1
                    else:
                        meaningful_parts.append(part)
                        i += 1
                        
                # Create a cleaner label from the meaningful parts
                if meaningful_parts:
                    clean_source = ' '.join(meaningful_parts).replace('_', ' ').title()
                else:
                    clean_source = source_name.replace('_', ' ').title()
            
            markdown_lines.append(f"> **{clean_source}:** {text.strip()}")
        elif text.strip():
            # If no specific source, just add the text
            markdown_lines.append(f"> {text.strip()}")
    
    # If no content texts were processed, just return an empty list
    if not markdown_lines:
        # Add a placeholder if there are chart elements but no text
        elements_dir = element.get('sub_image_path', '').replace('.jpg', '_elements') if element.get('sub_image_path') else None
        if elements_dir and os.path.exists(elements_dir):
            markdown_lines.append("> Chart elements detected but no text content extracted")
    
    return markdown_lines

def save_extracted_content_to_json(result_obj, extract_dir=None, output_file="extracted_content.json"):
    """
    Save the extracted content result object to a JSON file.
    
    Args:
        result_obj (dict): The result object from get_all_extracted_content
        extract_dir (str): Directory to save the JSON file. If None, saves in current directory
        output_file (str): Output file name (default: "extracted_content.json")
    """
    if extract_dir:
        output_path = os.path.join(extract_dir, output_file)
    else:
        output_path = output_file
    
    with open(output_path, "w") as f:
        json.dump(result_obj, f, indent=2)
    print(f"Extracted content saved to {output_path}")


def save_document_markdown(result_obj, extract_dir=None, source_fn=None):
    """
    Creates a markdown representation of the whole document's text extracts 
    and saves it to the extracts directory.
    
    Args:
        result_obj (dict): The result object from get_all_extracted_content
        extract_dir (str): Directory to save the markdown file. If None, uses default pattern
        source_fn (str): Source filename without extension for naming the markdown file
    """
    import os
    
    markdown_content = []
    
    # Add document title
    if source_fn:
        markdown_content.append(f"# {source_fn}\n")
    else:
        markdown_content.append(f"# Document\n")
    
    markdown_content.append("## Document Overview\n")
    markdown_content.append(f"- Total Pages: {len(result_obj.get('pages', {}))}")
    markdown_content.append(f"- Total Elements: {result_obj.get('total_elements', 0)}")
    
    # Add content statistics
    content_elements = result_obj.get('content_elements', {})
    markdown_content.append(f"- Tables: {len(content_elements.get('tables', []))}")
    markdown_content.append(f"- Charts: {len(content_elements.get('charts', []))}")
    markdown_content.append(f"- Titles: {len(content_elements.get('titles', []))}")
    markdown_content.append(f"- Other Elements: {len(content_elements.get('other', []))}\n")
    
    # Process each page in sorted order
    pages = result_obj.get('pages', {})
    for page_name in sorted(pages.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0):
        page_data = pages[page_name]
        
        markdown_content.append(f"## Page {page_name.replace('page_', '')}\n")
        
        # Add page text (from PDF extraction)
        page_text = page_data.get('page_text', '')
        if page_text.strip():
            markdown_content.append("### Page Text\n")
            markdown_content.append(f"{page_text}\n")
        
        # Process page elements
        elements = page_data.get('elements', [])
        if elements:
            markdown_content.append("### Page Elements\n")
            
            for element in elements:
                element_type = element.get('type', 'other')
                markdown_content.append(f"#### {element_type.title()}")
                
                # Add content texts if available
                content_texts = element.get('content_texts', [])
                if content_texts:
                    if element_type == 'table':
                        # Format table content using the new utility function
                        table_lines = format_markdown_table(element, content_texts)
                        for line in table_lines:
                            markdown_content.append(line)
                    elif element_type == 'chart':
                        # Format chart content using the new utility function
                        chart_lines = format_markdown_chart(element, content_texts)
                        for line in chart_lines:
                            markdown_content.append(line)
                    else:
                        # Handle non-chart, non-table content as before
                        for content in content_texts:
                            text = content.get('text', '')
                            if text.strip():
                                markdown_content.append(f"- {text.strip()}")
                
                markdown_content.append("")  # Extra blank line between elements
        
        markdown_content.append("---\n")  # Separator between pages
    
    # Join all content with proper spacing
    final_markdown = "\n".join(markdown_content)
    
    # Determine output path
    if extract_dir and source_fn:
        output_path = os.path.join(extract_dir, f"{source_fn}.md")
    elif extract_dir:
        # If extract_dir is provided but source_fn isn't, use a default name
        output_path = os.path.join(extract_dir, "document.md")
    else:
        # If no extract_dir provided, save in current directory
        output_path = f"{source_fn or 'document'}.md"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_markdown)
    
    print(f"Document markdown saved to {output_path}")


def generate_embeddings_for_markdown(markdown_file_path, api_key=None):
    """
    Generate embeddings for a markdown file, splitting content by page boundaries.
    Each page becomes a separate embedding chunk.

    Args:
        markdown_file_path (str): Path to the markdown file to process
        api_key (str): API key for NVIDIA embedding service. If not provided, 
                      will try to get from environment variable NVIDIA_API_KEY

    Returns:
        tuple: A tuple containing (results list, total time in seconds)
    """
    import os
    from openai import OpenAI
    import time
    
    start_time = time.time()
    
    # Set up API key
    if api_key is None:
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            print("NVIDIA_API_KEY environment variable not set. Skipping embeddings generation.")
            return [], 0.0
    
    # Initialize the client
    client = OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1"
    )

    # Read the markdown file
    with open(markdown_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split content by page separators (--- followed by a newline, which was added in save_document_markdown)
    sections = content.split('---\n')
    
    # Filter out empty sections and header sections
    filtered_sections = []
    for i, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        # Skip the header sections if they're not actual page content
        # Only skip sections that are purely document headers without page content
        if section.startswith('# ') and ('## Page' not in section):
            # This is likely just document headers without actual page content
            continue
        filtered_sections.append((i, section))
    
    # Use the new embedding batching function
    results = _make_embedding_batch_request(
        filtered_sections,
        client,
        batch_size=25,  # Reasonable batch size to stay within API limits
        api_description="embeddings"
    )
    
    total_time = time.time() - start_time
    print(f"Embedding generation completed in {total_time:.2f} seconds")
    
    return results, total_time


def save_embeddings_to_json(embedding_results, extract_dir=None, source_fn=None):
    """
    Save embedding results to a JSON file in the extracts directory.

    Args:
        embedding_results (list): List of embedding results from generate_embeddings_for_markdown
        extract_dir (str): Directory to save the embeddings file. If None, uses default pattern
        source_fn (str): Source filename without extension for naming the embeddings file
    """
    import json
    import os
    
    # Determine output path
    if extract_dir and source_fn:
        output_path = os.path.join(extract_dir, f"{source_fn}_embeddings.json")
    elif extract_dir:
        # If extract_dir is provided but source_fn isn't, use a default name
        output_path = os.path.join(extract_dir, "document_embeddings.json")
    else:
        # If no extract_dir provided, save in current directory
        output_path = f"{source_fn or 'document'}_embeddings.json"
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(embedding_results, f, indent=2)
    
    print(f"Embeddings saved to {output_path}")


def save_to_lancedb(embedding_results, extract_dir=None, source_fn=None):
    """
    Save embedding results to a LanceDB table for queryable storage.

    Args:
        embedding_results (list): List of embedding results from generate_embeddings_for_markdown
        extract_dir (str): Directory to save the LanceDB database. If None, uses default pattern
        source_fn (str): Source filename without extension for naming the database
    """
    import lancedb
    import os
    import pyarrow as pa
    import time
    
    start_time = time.time()
    
    try:
        # Determine database path
        if extract_dir:
            db_path = os.path.join(extract_dir, "lancedb")
        else:
            db_path = "./lancedb"
        
        # Connect to LanceDB
        db = lancedb.connect(db_path)
        
        # Prepare data for insertion
        # Only include results that have successful embeddings
        valid_results = [r for r in embedding_results if r.get('embedding') is not None]
        
        if not valid_results:
            print("No valid embeddings to store in LanceDB")
            indexing_time = time.time() - start_time
            return None, indexing_time
        
        # Create schema for the table
        # Using PyArrow schema to define table structure
        schema = pa.schema([
            pa.field("page_index", pa.int32()),
            pa.field("content", pa.string()),
            pa.field("embedding", pa.list_(pa.float32())),  # Embedding vector
            pa.field("source_document", pa.string()),      # Name of the source document
            pa.field("page_content_length", pa.int32())    # Length of content for metadata
        ])
        
        # Prepare data for insertion
        data = []
        for result in valid_results:
            data.append({
                "page_index": result["page_index"],
                "content": result["content"],
                "embedding": result["embedding"],
                "source_document": source_fn or "unknown",
                "page_content_length": len(result["content"])
            })
        
        # Convert to PyArrow table
        table_data = pa.Table.from_pylist(data, schema=schema)
        
        # Use a single table for all documents
        table_name = "all_documents"
        
        if table_name in db.table_names():
            # If table exists, open it and add the new data
            table = db.open_table(table_name)
            table.add(table_data)
        else:
            # Create new table
            table = db.create_table(table_name, table_data)
        
        indexing_time = time.time() - start_time
        print(f"Successfully saved {len(valid_results)} embeddings to LanceDB table '{table_name}' in {db_path}")
        print(f"LanceDB table has {table.count_rows()} total rows")
        print(f"LanceDB indexing completed in {indexing_time:.2f} seconds")
        
        return table_name, indexing_time
        
    except ImportError:
        print("LanceDB not installed. Install with: pip install lancedb")
        indexing_time = time.time() - start_time
        return None, indexing_time
    except Exception as e:
        print(f"Error saving to LanceDB: {str(e)}")
        indexing_time = time.time() - start_time
        return None, indexing_time
    
