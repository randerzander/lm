import os
import json
import time
import requests
import base64
from glob import glob
from PIL import Image
import concurrent.futures
from threading import Lock, Semaphore
import threading
from collections import deque
import datetime

# Global variable to track the last log time
_last_log_time = None

def log(message, level="DEBUG"):
    """
    Log a message with current timestamp and time elapsed since the last log call.
    Format: [YYYY-MM-DD HH:MM:SS] message (+N secs) for DEBUG level
    For ALWAYS level: just print the message without timestamp or elapsed time
    
    Only prints DEBUG messages if global DEBUG mode is enabled.
    ALWAYS level messages are printed regardless of DEBUG mode.
    """
    global _last_log_time
    
    # Check if DEBUG mode is enabled
    debug_enabled = os.environ.get('DEBUG', '').lower() in ('true', '1', 'yes', 'on', 'debug')
    
    # Only print DEBUG level messages if debug is enabled, always print ALWAYS level
    if level == "ALWAYS":
        # For ALWAYS level, just print the message without timestamp or elapsed time
        print(message)
    elif level == "DEBUG" and debug_enabled:
        current_time = time.time()
        current_datetime = datetime.datetime.fromtimestamp(current_time)
        timestamp_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        
        if _last_log_time is not None:
            elapsed = current_time - _last_log_time
            log_message = f"[{timestamp_str}] {message} (+{elapsed:.1f} secs)"
        else:
            log_message = f"[{timestamp_str}] {message}"
        
        print(log_message)
        _last_log_time = current_time

# Global rate limiter for API requests to prevent exceeding rate limits (default: 10 concurrent requests)
_api_rate_limit_semaphore = Semaphore(10)  # Default rate limit: 10 concurrent requests
_api_request_timestamps = deque()  # Track request timestamps for rate monitoring
_api_rate_limit_lock = Lock()  # Protect access to timestamp tracking
_api_max_concurrent_requests = 10  # Configurable maximum concurrent requests
_api_requests_per_minute = 40  # Default: 40 requests per minute
_api_max_workers = None  # Default to system CPU count for thread pools

def configure_api_rate_limit(max_concurrent_requests=10, requests_per_minute=40, max_workers=None):
    """
    Configure the API rate limiting parameters.
    
    Args:
        max_concurrent_requests (int): Maximum number of concurrent API requests allowed (default: 10)
        requests_per_minute (int): Maximum number of requests allowed per minute (default: 40)
        max_workers (int, optional): Maximum number of workers for thread pools (default: None, uses system CPU count)
    """
    global _api_rate_limit_semaphore, _api_max_concurrent_requests, _api_requests_per_minute, _api_max_workers
    
    with _api_rate_limit_lock:
        _api_max_concurrent_requests = max_concurrent_requests
        _api_requests_per_minute = requests_per_minute
        _api_max_workers = max_workers
        # Create a new semaphore with the updated limit
        _api_rate_limit_semaphore = Semaphore(max_concurrent_requests)
        
    log(f"API rate limit configured: {max_concurrent_requests} concurrent requests maximum, {requests_per_minute} requests per minute maximum")
    if max_workers:
        log(f"Thread pool max workers configured: {max_workers}")

def _enforce_rate_limit():
    """
    Enforce rate limiting based on time window to prevent exceeding requests per minute.
    """
    current_time = time.time()
    
    with _api_rate_limit_lock:
        # Remove timestamps older than 60 seconds
        while _api_request_timestamps and (current_time - _api_request_timestamps[0]) > 60:
            _api_request_timestamps.popleft()
        
        # If we've reached the limit, wait until we can make another request
        if len(_api_request_timestamps) >= _api_requests_per_minute:
            # Calculate how long to wait until the oldest request is outside the window
            oldest_time = _api_request_timestamps[0]
            wait_time = 60 - (current_time - oldest_time) + 0.1  # Add small buffer
            return wait_time
    
    return 0  # No waiting needed

# Global rate limiter for API requests to prevent exceeding rate limits
api_request_lock = _api_rate_limit_semaphore

def _make_batch_request(items, api_endpoint, headers, batch_size, payload_processor, result_processor, api_description="batch", max_workers=None, parallel=False):
    """
    Generic function for making batch requests to an API with rate limiting.
    
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
    
    # Use configured max_workers if not provided
    if max_workers is None:
        max_workers = _api_max_workers
    
    # Create all batches first
    all_batches = []
    for i in range(0, len(items), batch_size):
        batch_items = items[i:i + batch_size]
        all_batches.append(batch_items)
    
    if parallel and len(all_batches) > 1:
        # Process all batches in parallel
        log(f"Processing {len(all_batches)} {api_description} batches in parallel...")
        
        def process_single_batch(batch_idx_batch_items):
            batch_idx, batch_items = batch_idx_batch_items
            log(f"Processing {api_description} batch {batch_idx + 1}/{len(all_batches)} ({len(batch_items)} items)")  # Commented out to reduce noise
            log(f"  Items in batch: {[os.path.basename(item) if isinstance(item, str) else item for item in batch_items]}")  # Commented out to reduce noise
            
            # Prepare batch payload using the provided processor
            payload = payload_processor(batch_items)
            
            try:
                # Use rate limiter to prevent exceeding API rate limits (concurrent + time-based)
                with api_request_lock:
                    # Enforce time-based rate limiting
                    wait_time = _enforce_rate_limit()
                    if wait_time > 0:
                        time.sleep(wait_time)
                    
                    # Record request timestamp
                    with _api_rate_limit_lock:
                        _api_request_timestamps.append(time.time())
                    
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
            # print(f"Processing {api_description} batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1} ({len(batch_items)} items)")  # Commented out to reduce noise
            # print(f"  Items in batch: {[os.path.basename(item) if isinstance(item, str) else item for item in batch_items]}")  # Commented out to reduce noise
            
            # Prepare batch payload using the provided processor
            payload = payload_processor(batch_items)
            
            try:
                # Use rate limiter to prevent exceeding API rate limits (concurrent + time-based)
                with api_request_lock:
                    # Enforce time-based rate limiting
                    wait_time = _enforce_rate_limit()
                    if wait_time > 0:
                        time.sleep(wait_time)
                    
                    # Record request timestamp
                    with _api_rate_limit_lock:
                        _api_request_timestamps.append(time.time())
                    
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


def calculate_smart_batch_size(file_paths, max_batch_size=25, max_total_payload_size=2_000_000):
    """
    Calculate an appropriate batch size based on both element count and total payload size.
    
    Args:
        file_paths (list): List of file paths to process
        max_batch_size (int): Maximum number of elements per batch (default: 25)
        max_total_payload_size (int): Maximum total payload size in bytes (default: 2MB)
        
    Returns:
        int: Smart batch size that respects both limits
    """
    import os
    
    if not file_paths:
        return max_batch_size
    
    # Estimate payload size by sampling first few files
    sample_size = min(5, len(file_paths))
    total_sample_size = 0
    
    for i in range(sample_size):
        try:
            file_size = os.path.getsize(file_paths[i])
            # Rough estimate: base64 encoding increases size by ~33%, plus JSON overhead
            estimated_encoded_size = int(file_size * 1.4)
            total_sample_size += estimated_encoded_size
        except OSError:
            # If file doesn't exist or can't be accessed, use a conservative estimate
            total_sample_size += 100_000  # 100KB estimate
    
    # Calculate average size per file based on samples
    avg_file_size = total_sample_size // sample_size if sample_size > 0 else 150_000
    
    # Calculate batch size based on payload size limit
    max_batch_by_size = max_total_payload_size // avg_file_size
    max_batch_by_size = max(1, max_batch_by_size)  # Ensure at least 1
    
    # Return the most restrictive limit
    smart_batch_size = min(max_batch_size, max_batch_by_size)
    
    log(f"Smart batch sizing: {len(file_paths)} files, avg size ~{avg_file_size//1000}KB, "
          f"batch size: {smart_batch_size} (was {max_batch_size})")
    
    return smart_batch_size


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
        log(f"Processing {api_description} batch {i//batch_size + 1}/{(len(items)-1)//batch_size + 1} ({len(batch_items)} items)")
        
        # Prepare batch content
        batch_contents = [item[1] for item in batch_items]  # item[1] is the content
        
        try:
            # Use rate limiter to prevent exceeding API rate limits (concurrent + time-based)
            with api_request_lock:
                # Enforce time-based rate limiting
                wait_time = _enforce_rate_limit()
                if wait_time > 0:
                    time.sleep(wait_time)
                
                # Record request timestamp
                with _api_rate_limit_lock:
                    _api_request_timestamps.append(time.time())
            
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
                
        except Exception as e:
            print(f"Error processing {api_description} batch starting at item {batch_items[0][0]}: {str(e)}")
            # Fallback: process individually if batch fails
            for page_idx, content in batch_items:
                try:
                    # Use rate limiter to prevent exceeding API rate limits (concurrent + time-based)
                    with api_request_lock:
                        # Enforce time-based rate limiting
                        wait_time = _enforce_rate_limit()
                        if wait_time > 0:
                            time.sleep(wait_time)
                        
                        # Record request timestamp
                        with _api_rate_limit_lock:
                            _api_request_timestamps.append(time.time())
                    
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
                    
                except Exception as e_single:
                    print(f"Error generating embedding for item {page_idx}: {str(e_single)}")
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

    # print(f"Processing individual page element inference for: {image_path}")  # Commented out to reduce noise
    # Use rate limiter to prevent exceeding API rate limits (concurrent + time-based)
    with api_request_lock:
        # Enforce time-based rate limiting
        wait_time = _enforce_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Record request timestamp
        with _api_rate_limit_lock:
            _api_request_timestamps.append(time.time())
        
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

    # print(f"Processing individual table structure inference for: {image_path}")  # Commented out to reduce noise
    # Use rate limiter to prevent exceeding API rate limits (concurrent + time-based)
    with api_request_lock:
        # Enforce time-based rate limiting
        wait_time = _enforce_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Record request timestamp
        with _api_rate_limit_lock:
            _api_request_timestamps.append(time.time())
        
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

    # print(f"Processing individual OCR inference for: {image_path}")  # Commented out to reduce noise
    # Use rate limiter to prevent exceeding API rate limits (concurrent + time-based)
    with api_request_lock:
        # Enforce time-based rate limiting
        wait_time = _enforce_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Record request timestamp
        with _api_rate_limit_lock:
            _api_request_timestamps.append(time.time())
        
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

    # print(f"Processing individual graphic elements inference for: {image_path}")  # Commented out to reduce noise
    # Use rate limiter to prevent exceeding API rate limits (concurrent + time-based)
    with api_request_lock:
        # Enforce time-based rate limiting
        wait_time = _enforce_rate_limit()
        if wait_time > 0:
            time.sleep(wait_time)
        
        # Record request timestamp
        with _api_rate_limit_lock:
            _api_request_timestamps.append(time.time())
        
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

def process_page_images(pages_dir="pages", output_dir="page_elements", timing=False, ocr_titles=True, batch_processing=True, batch_size=25, pdf_extraction_time=0):
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
        log(f"Processing {len(page_images)} page images using batch processing...", level="ALWAYS")
        
        # Process page images in batches for page element detection
        for i in range(0, len(page_images), batch_size):
            batch_paths = page_images[i:i + batch_size]
            # print(f"DEBUG: Processing batch {i//batch_size + 1} with {len(batch_paths)} images: {batch_paths[:2]}{'...' if len(batch_paths) > 2 else ''}")  # Show first 2 paths for debugging
            
            if timing:
                start_time = time.time()
            try:
                # Use smart batching that considers both element count and total payload size
                smart_batch_size = calculate_smart_batch_size(batch_paths, max_batch_size=batch_size, max_total_payload_size=2_000_000)  # 2MB limit
                batch_results = extract_bounding_boxes_batch(batch_paths, api_key, smart_batch_size)
                
                if timing:
                    page_elements_time += time.time() - start_time
                
                # DEBUG: Add diagnostic information for batch processing
                # print(f"DEBUG: Batch results received - len(batch_results)={len(batch_results)}, batch_paths_count={len(batch_paths)}")
                
                # DEBUG: Inspect what batch_results contains
                # if len(batch_results) > 0:
                #     print(f"DEBUG: First batch_result type: {type(batch_results[0])}")
                #     if isinstance(batch_results[0], dict) and 'data' in batch_results[0]:
                #         print(f"DEBUG: First batch_result data keys: {list(batch_results[0]['data'].keys()) if isinstance(batch_results[0]['data'], dict) else type(batch_results[0]['data'])}")
                #     elif hasattr(batch_results[0], '__iter__') and not isinstance(batch_results[0], str):
                #         try:
                #             print(f"DEBUG: First batch_result is iterable with {len(list(batch_results[0]))} items")
                #         except:
                #             print(f"DEBUG: First batch_result is iterable but length unknown")
                
                # Process the batch results
                # FIX: extract_bounding_boxes_batch returns results for the CURRENT batch (single batch),
                # not a list of all batches processed so far.
                # So we always take index 0, not i // batch_size
                if len(batch_results) == 0:
                    print(f"ERROR: No batch results returned for batch with {len(batch_paths)} images")
                    raise ValueError(f"No batch results returned for batch with {len(batch_paths)} images")
                
                batch_result = batch_results[0]  # Get the result for the current batch
                # print(f"DEBUG: Successfully retrieved batch_result at index 0 (current batch)")
                
                # The API response structure has 'data' field which contains page data for each image in the batch
                if 'data' in batch_result:
                    batch_page_data = batch_result['data']
                    
                    for img_idx, img_path in enumerate(batch_paths):
                        # print(f"Processing {img_path}...")  # Commented out to reduce noise
                        if img_idx < len(batch_page_data):
                            page_data = batch_page_data[img_idx]
                        else:
                            # print(f"No data found in response for {img_path}")  # Commented out to reduce noise
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
                # DEBUG: Add more detailed error information
                # import traceback
                # print(f"DEBUG: Full traceback for batch error:")
                # traceback.print_exc()
                
                # If batch processing fails, fall back to individual processing
                for img_path in batch_paths:
                    # print(f"Falling back to processing {img_path} individually...")  # Commented out to reduce noise
                    # We would need to call single image processing here
                    pass
    else:
        # Process page images individually if batch processing is disabled
        for image_path in sorted(page_images):
            # print(f"Processing {image_path}...")  # Commented out to reduce noise
            
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
        log(f"Processing {len(table_structure_tasks)} table structure tasks in parallel batches...", level="ALWAYS")
        
        # Extract the temporary image paths to process in batch
        temp_paths = [task['temp_path'] for task in table_structure_tasks]
        
        try:
            if timing:
                start_time = time.time()
            # Process all table structure batches in parallel with smart sizing
            smart_batch_size = calculate_smart_batch_size(temp_paths, max_batch_size=batch_size, max_total_payload_size=2_000_000)  # 2MB limit
            table_structure_batch_results = extract_table_structure_batch(temp_paths, api_key, smart_batch_size)
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
        log(f"Processing {len(chart_graphic_elements_tasks)} chart graphic elements tasks in parallel batches...", level="ALWAYS")
        
        # Extract the temporary image paths to process in batch
        temp_paths = [task['temp_path'] for task in chart_graphic_elements_tasks]
        
        try:
            if timing:
                start_time = time.time()
            # Process all chart graphic elements batches in parallel with smart sizing
            smart_batch_size = calculate_smart_batch_size(temp_paths, max_batch_size=batch_size, max_total_payload_size=2_000_000)  # 2MB limit
            chart_graphic_elements_batch_results = extract_graphic_elements_batch(temp_paths, api_key, smart_batch_size)
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

    # Process OCR tasks in batches with smart sizing
    if all_ocr_tasks:
        # Use smart batching that considers both element count and total payload size
        smart_batch_size = calculate_smart_batch_size(all_ocr_tasks, max_batch_size=batch_size, max_total_payload_size=2_000_000)  # 2MB limit
        log(f"Processing {len(all_ocr_tasks)} OCR tasks in parallel batches (smart batch size: {smart_batch_size})...", level="ALWAYS")
        try:
            if timing:
                start_time = time.time()
            # Process all OCR batches in parallel
            ocr_batch_results = extract_ocr_text_batch(all_ocr_tasks, api_key, smart_batch_size, parallel=True)
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
                                # Try to find structure path based on sub_image_path if structure_path field doesn't exist
                                structure_path = element_data.get('structure_path')
                                if not structure_path and 'sub_image_path' in element_data and element_data['sub_image_path']:
                                    potential_structure_path = element_data['sub_image_path'].replace('.jpg', '_structure.json').replace('.png', '_structure.json')
                                    if os.path.exists(potential_structure_path):
                                        structure_path = potential_structure_path
                                
                                if structure_path and os.path.exists(structure_path):
                                    with open(structure_path, 'r') as struct_file:
                                        struct_data = json.load(struct_file)
                                        
                                        # Add table structure info
                                        element_record['table_structure_path'] = structure_path
                                        
                                        # Process cells from the bounding_boxes directly (not from a 'data' array)
                                        if 'bounding_boxes' in struct_data and 'cell' in struct_data['bounding_boxes']:
                                            cells = struct_data['bounding_boxes']['cell']
                                            
                                            # Get cell images and OCR text
                                            cells_dir = element_data['sub_image_path'].replace('.jpg', '_cells').replace('.png', '_cells')
                                            if os.path.exists(cells_dir):
                                                # Process all potential cell files based on the structure data
                                                for cell_idx, cell in enumerate(cells):
                                                    cell_image_path = os.path.join(cells_dir, f"{os.path.basename(element_data['sub_image_path']).replace('.jpg', '').replace('.png', '')}_cell_{cell_idx+1}.jpg")
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
                                
                                # Also try to find text content from the main table image OCR
                                if 'sub_image_path' in element_data and element_data['sub_image_path']:
                                    table_ocr_path = element_data['sub_image_path'].replace('.jpg', '_ocr.json').replace('.png', '_ocr.json')
                                    if os.path.exists(table_ocr_path):
                                        with open(table_ocr_path, 'r') as ocr_file:
                                            ocr_data = json.load(ocr_file)
                                            if 'data' in ocr_data and ocr_data['data']:
                                                for ocr_item in ocr_data['data']:
                                                    if 'text_detections' in ocr_item:
                                                        for text_det in ocr_item['text_detections']:
                                                            element_record['content_texts'].append({
                                                                'text': text_det['text_prediction']['text'],
                                                                'confidence': text_det['text_prediction']['confidence'],
                                                                'source': 'table_main',
                                                                'bounding_box': text_det.get('bounding_box')
                                                            })
                                
                                # Add to tables list
                                result['content_elements']['tables'].append(element_record)
                                
                            elif content_type == 'chart':
                                # Try to find elements path based on sub_image_path if elements_path field doesn't exist
                                elements_path = element_data.get('elements_path')
                                if not elements_path and 'sub_image_path' in element_data and element_data['sub_image_path']:
                                    potential_elements_path = element_data['sub_image_path'].replace('.jpg', '_elements.json').replace('.png', '_elements.json')
                                    if os.path.exists(potential_elements_path):
                                        elements_path = potential_elements_path
                                
                                # Add chart elements with fallback content extraction
                                if elements_path and os.path.exists(elements_path):
                                    element_record['chart_elements_path'] = elements_path
                                    
                                    # Get chart element images and OCR
                                    elements_dir = element_data['sub_image_path'].replace('.jpg', '_elements').replace('.png', '_elements')
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
                                
                                # Also try to find text content from the main chart image OCR
                                if 'sub_image_path' in element_data and element_data['sub_image_path']:
                                    chart_ocr_path = element_data['sub_image_path'].replace('.jpg', '_ocr.json').replace('.png', '_ocr.json')
                                    if os.path.exists(chart_ocr_path):
                                        with open(chart_ocr_path, 'r') as ocr_file:
                                            ocr_data = json.load(ocr_file)
                                            if 'data' in ocr_data and ocr_data['data']:
                                                for ocr_item in ocr_data['data']:
                                                    if 'text_detections' in ocr_item:
                                                        for text_det in ocr_item['text_detections']:
                                                            element_record['content_texts'].append({
                                                                'text': text_det['text_prediction']['text'],
                                                                'confidence': text_det['text_prediction']['confidence'],
                                                                'source': 'chart_main',
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
            # The structure is direct in the file, not in a 'data' array
            if 'bounding_boxes' in struct_data and 'cell' in struct_data['bounding_boxes']:
                cells = struct_data['bounding_boxes']['cell']

                # Now we'll use both the structure information and the OCR content
                # to build a proper markdown table

                # For a proper markdown table, we need to determine the grid structure
                # from the cell coordinates. Since this is complex, we'll implement
                # a simplified version that attempts to organize cells into rows.

                # First, let's get the cell data with coordinates if available
                # and the actual OCR text content
                cell_data_with_coords = []
                # Match cells from structure file with OCR content texts by index
                # Only process cells that have corresponding OCR content
                for i, (cell_id, text) in enumerate(sorted_cells):
                    if i < len(cells):  # Only process if we have a corresponding structure cell
                        cell_info = cells[i]
                        # Use the structure coordinates and the OCR text
                        cell_data_with_coords.append({
                            'text': text,
                            'x_min': cell_info.get('x_min', 0),
                            'y_min': cell_info.get('y_min', 0),
                            'x_max': cell_info.get('x_max', 1),
                            'y_max': cell_info.get('y_max', 1)
                        })

                # If we have cell data, try to create a proper table structure
                # This should be OUTSIDE the loop that builds cell_data_with_coords
                if cell_data_with_coords:
                    # For simplicity of this implementation, we'll sort cells by y-coordinate first
                    # (for rows) and then by x-coordinate (for columns) to simulate table structure
                    # This is a simplified approach - a robust implementation would use
                    # more sophisticated algorithms to determine table grid

                    # Group cells by similar y-coordinates (rows)
                    # Use a simple approach with tolerance for y-coordinates
                    row_tolerance = 0.05  # Adjust based on coordinate system
                    rows = []
                    used_cell_indices = set()  # Track indices of cells we've already used

                    # Create a list of cell data with their original indices to properly track them
                    indexed_cell_data = [(i, cell) for i, cell in enumerate(cell_data_with_coords)]

                    for idx, cell in indexed_cell_data:
                        if idx in used_cell_indices:
                            continue

                        current_row = [cell]
                        used_cell_indices.add(idx)

                        # Find other cells with similar y_min (in the same row)
                        for other_idx, other_cell in indexed_cell_data:
                            if other_idx in used_cell_indices:
                                continue
                            if abs(cell['y_min'] - other_cell['y_min']) < row_tolerance:
                                current_row.append(other_cell)
                                used_cell_indices.add(other_idx)

                        # Sort cells in row by x-coordinate (left to right)
                        current_row.sort(key=lambda c: c['x_min'])
                        rows.append(current_row)

                    # Now sort rows by y-coordinate (top to bottom)
                    rows.sort(key=lambda r: r[0]['y_min'] if r else 0)

                    # Create markdown table - only once
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
    
    # Process each page in sorted order
    pages = result_obj.get('pages', {})
    for page_name in sorted(pages.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0):
        page_data = pages[page_name]
        
        # Add page text (from PDF extraction)
        page_text = page_data.get('page_text', '')
        if page_text.strip():
            markdown_content.append(f"{page_text}\n")
        
        # Process page elements
        elements = page_data.get('elements', [])
        #print(f"Got {len(elements)} elements for page {page_name}")
        if elements:
            for element in elements:
                element_type = element.get('type', 'other')
                markdown_content.append(f"#### {element_type.title()}")
                
                # Add content texts if available
                content_texts = element.get('content_texts', [])
                #print(f"Element has {len(content_texts)} content texts")
                if content_texts:
                    if element_type == 'table':
                        # Format table content using the new utility function
                        #print(f"Adding table from page {page_name}")
                        table_lines = format_markdown_table(element, content_texts)
                        #print(table_lines)
                        for line in table_lines:
                            markdown_content.append(line)
                    elif element_type == 'chart':
                        # Format chart content using the new utility function
                        #print(f"Adding chart from page {page_name}")
                        chart_lines = format_markdown_chart(element, content_texts)
                        #print(chart_lines)
                        for line in chart_lines:
                            markdown_content.append(line)
                    else:
                        # Handle non-chart, non-table content as before
                        #print(f"Adding {element_type} from page {page_name}")
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
    
    log(f"Document markdown saved to {output_path}")



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
    
    # Use a single database in the project root for all documents
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
        pa.field("page_number", pa.string()),          # Page number where content was located
        pa.field("content", pa.string()),
        pa.field("embedding", pa.list_(pa.float32())),  # Embedding vector
        pa.field("source_document", pa.string()),      # Name of the source document
        pa.field("page_content_length", pa.int32()),   # Length of content for metadata
        pa.field("content_type", pa.string())          # Type of content (page_text, chart, table, etc.)
    ])
    
    # Prepare data for insertion
    data = []
    for result in valid_results:
        typ = result.get("element_type", "unknown")
        cont = result["content"]
        #print(f"{typ} {cont[0:10]}..., len: {len(cont)}")
        data.append({
            "page_number": result.get("page_number", "unknown"),  # Use actual page number where content was located
            "content": result["content"],
            "embedding": result["embedding"],
            "source_document": source_fn or "unknown",
            "page_content_length": len(result["content"]),
            "content_type": result.get("element_type", "unknown")  # Use element_type from result metadata
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
    log(f"Successfully saved {len(valid_results)} embeddings to LanceDB table '{table_name}' in {db_path}")
    log(f"LanceDB table has {table.count_rows()} total rows")
    log(f"LanceDB indexing completed in {indexing_time:.2f} seconds", level="ALWAYS")
    
    return table_name, indexing_time

def generate_embeddings_from_result(result_obj, api_key=None):
    """
    Generate embeddings directly from the result object with granular chunks 
    for page texts, tables, and charts instead of full pages.
    
    Args:
        result_obj (dict): The result object from get_all_extracted_content
        api_key (str): API key for NVIDIA embedding service. If not provided, 
                      will try to get from environment variable NVIDIA_API_KEY

    Returns:
        tuple: A tuple containing (results list, total time in seconds)
    """
    import os
    from openai import OpenAI
    import time
    import json
    
    start_time = time.time()
    
    # Initialize the client
    client = OpenAI(
        api_key=os.environ["NVIDIA_API_KEY"],
        base_url="https://integrate.api.nvidia.com/v1"
    )

    # Prepare content chunks for embedding
    content_chunks = []
    
    # Process each page in sorted order
    pages = result_obj.get('pages', {})
    for page_name in sorted(pages.keys(), key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0):
        page_data = pages[page_name]
        
        # Add page text as a separate chunk if it exists
        page_text = page_data.get('page_text', '')
        if page_text.strip():
            # Wrap page text in markdown format (e.g., as a section)
            formatted_page_text = '# Page ' + page_name.replace('page_', '') + '\n\n' + page_text.strip()
            content_chunks.append({
                'type': 'page_text',
                'page_name': page_name,
                'content': formatted_page_text,
                'element_type': 'page_text',
                'source': page_name + '_page_text'
            })
        
        # Process page elements (tables, charts, titles, etc.)
        elements = page_data.get('elements', [])
        for element_idx, element in enumerate(elements):
            element_type = element.get('type', 'other')
            content_texts = element.get('content_texts', [])
            
            # For tables, embed their markdown formatted content
            if element_type == 'table' and content_texts:
                # Format table content using the markdown formatting function
                try:
                    table_lines = format_markdown_table(element, content_texts)
                    formatted_content = '\n'.join(table_lines) if table_lines else ''
                    
                    if formatted_content.strip():
                        content_chunks.append({
                            'type': element_type,
                            'page_name': page_name,
                            'content': formatted_content,
                            'element_type': element_type,
                            'source': page_name + '_' + element_type + '_' + str(element_idx+1)
                        })
                except:
                    # Fallback if format_markdown_table fails
                    element_content_parts = []
                    for content in content_texts:
                        text = content.get('text', '')
                        if text.strip():
                            element_content_parts.append(text.strip())
                    if element_content_parts:
                        combined_content = ' '.join(element_content_parts)
                        content_chunks.append({
                            'type': element_type,
                            'page_name': page_name,
                            'content': combined_content,
                            'element_type': element_type,
                            'source': page_name + '_' + element_type + '_' + str(element_idx+1)
                        })
            
            # For charts, embed their markdown formatted content
            elif element_type == 'chart' and content_texts:
                # Format chart content using the markdown formatting function
                try:
                    chart_lines = format_markdown_chart(element, content_texts)
                    formatted_content = '\n'.join(chart_lines) if chart_lines else ''
                    
                    if formatted_content.strip():
                        content_chunks.append({
                            'type': element_type,
                            'page_name': page_name,
                            'content': formatted_content,
                            'element_type': element_type,
                            'source': page_name + '_' + element_type + '_' + str(element_idx+1)
                        })
                except:
                    # Fallback if format_markdown_chart fails
                    element_content_parts = []
                    for content in content_texts:
                        text = content.get('text', '')
                        if text.strip():
                            element_content_parts.append(text.strip())
                    if element_content_parts:
                        combined_content = ' '.join(element_content_parts)
                        content_chunks.append({
                            'type': element_type,
                            'page_name': page_name,
                            'content': combined_content,
                            'element_type': element_type,
                            'source': page_name + '_' + element_type + '_' + str(element_idx+1)
                        })
            
            # For other elements (excluding titles), embed their content as markdown formatted text
            elif content_texts and element_type != 'title':
                # Format other elements as markdown text (similar to how save_document_markdown does it)
                element_content_parts = []
                for content in content_texts:
                    text = content.get('text', '')
                    if text.strip():
                        element_content_parts.append('- ' + text.strip())
                
                if element_content_parts:
                    combined_content = '\\n'.join(element_content_parts)
                    content_chunks.append({
                        'type': element_type,
                        'page_name': page_name,
                        'content': combined_content,
                        'element_type': element_type,
                        'source': page_name + '_' + element_type + '_' + str(element_idx+1)
                    })
    
    # Create section tuples for batching (index, content format that _make_embedding_batch_request expects)
    sections = [(i, chunk['content']) for i, chunk in enumerate(content_chunks)]
    
    # Use the new embedding batching function
    # For embeddings, use a fixed reasonable batch size since content varies significantly
    batch_size = 25  # Reasonable batch size to stay within API limits
    embedding_results = _make_embedding_batch_request(
        sections,
        client,
        batch_size=batch_size,
        api_description="embeddings"
    )
    
    # Enhance results with additional metadata from content_chunks
    for i, result in enumerate(embedding_results):
        if i < len(content_chunks) and result.get('embedding') is not None:
            chunk = content_chunks[i]
            result['page_name'] = chunk['page_name']
            result['element_type'] = chunk['element_type']
            result['source'] = chunk['source']
            # Extract the page number from page_name (e.g., 'page_001' -> '1')
            page_name_raw = chunk['page_name'].replace('page_', '') if chunk['page_name'].startswith('page_') else chunk['page_name']
            # Remove leading zeros but preserve "0" for page 0
            page_number = str(int(page_name_raw)) if page_name_raw.isdigit() else page_name_raw
            result['page_number'] = page_number  # Use actual page number where content was located
            result['content'] = chunk['content']  # Ensure content is set
    
    total_time = time.time() - start_time
    return embedding_results, total_time
