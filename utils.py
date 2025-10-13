import os
import json
import time
import requests
import base64
from glob import glob
from PIL import Image

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

    results = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"Processing page elements batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1} ({len(batch_paths)} images)")
        print(f"  Images in batch: {batch_paths}")
        
        # Prepare batch payload
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
        
        payload = {"input": inputs}
        
        try:
            response = requests.post(invoke_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                batch_result = response.json()
                results.append(batch_result)
            else:
                print(f"Page elements API request failed with status {response.status_code}: {response.text}")
                # Return partial results or raise exception based on requirements
                raise requests.exceptions.RequestException(f"Page elements API request failed: {response.status_code}")
        except Exception as e:
            print(f"Error processing page elements batch: {str(e)}")
            # Continue with other batches or raise exception based on requirements
            raise
    
    return results


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

    results = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"Processing table structure batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1} ({len(batch_paths)} images)")
        print(f"  Images in batch: {batch_paths}")
        
        # Prepare batch payload
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
        
        payload = {"input": inputs}
        
        try:
            response = requests.post(invoke_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                batch_result = response.json()
                results.append(batch_result)
            else:
                print(f"Table structure API request failed with status {response.status_code}: {response.text}")
                # Return partial results or raise exception based on requirements
                raise requests.exceptions.RequestException(f"Table structure API request failed: {response.status_code}")
        except Exception as e:
            print(f"Error processing table structure batch: {str(e)}")
            # Continue with other batches or raise exception based on requirements
            raise
    
    return results





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


def extract_ocr_text_batch(image_paths, api_key=None, batch_size=20):
    """
    Extract text from multiple images using NVIDIA OCR API in batches.
    
    Args:
        image_paths (list): List of paths to image files
        api_key (str, optional): API key for authorization. If not provided, 
                                assumes running in NGC environment
        batch_size (int): Number of images to process in each batch (default: 20)
    
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

    results = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        print(f"Processing OCR batch {i//batch_size + 1}/{(len(image_paths)-1)//batch_size + 1} ({len(batch_paths)} images)")
        print(f"  Images in batch: {batch_paths}")
        
        # Prepare batch payload
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
        
        payload = {"input": inputs}
        
        try:
            response = requests.post(invoke_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                batch_result = response.json()
                results.append(batch_result)
            else:
                print(f"OCR API request failed with status {response.status_code}: {response.text}")
                # Return partial results or raise exception based on requirements
                raise requests.exceptions.RequestException(f"OCR API request failed: {response.status_code}")
        except Exception as e:
            print(f"Error processing OCR batch: {str(e)}")
            # Continue with other batches or raise exception based on requirements
            raise
    
    return results





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

def process_page_images(pages_dir="pages", output_dir="page_elements", timing=False, ocr_titles=True, batch_processing=True, batch_size=20, pdf_extraction_time=0):
    """
    Process all page images in the specified directory, extract content elements,
    and save them in subdirectories organized by content type in JSONL format.
    Uses batch processing for page element extraction with fallback to single image processing.
    
    Args:
        pages_dir (str): Directory containing page images
        output_dir (str): Output directory for extracted elements
        timing (bool): Whether to track and report timing for each stage
        ocr_titles (bool): Whether to perform OCR on title elements, defaults to True
        batch_processing (bool): Whether to use batch processing for API calls (default: True)
        batch_size (int): Batch size for API calls (default: 20)
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
    
    # Lists to collect OCR tasks for batch processing later
    table_cell_ocr_tasks = []
    chart_element_ocr_tasks = []
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
                        
                        # Print the result structure for debugging
                        print(f"API response structure: {list(batch_result.keys()) if isinstance(batch_result, dict) else type(batch_result)}")
                        
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
                                    
                                    # If this is a table, also extract table structure
                                    if content_type == 'table':
                                        # Save the cropped image temporarily for API call
                                        temp_table_path = image_filename.replace('.jpg', '_for_api.jpg')
                                        cropped_image.save(temp_table_path, "JPEG", quality=80)
                                        
                                        # Extract table structure using the cropped image
                                        try:
                                            if timing:
                                                start_time = time.time()
                                            table_structure = extract_table_structure(temp_table_path, api_key)
                                            if timing:
                                                table_structure_time += time.time() - start_time
                                            
                                            # Save the table structure as a JSON file
                                            structure_filename = image_filename.replace('.jpg', '_structure.json')
                                            with open(structure_filename, 'w') as f:
                                                json.dump(table_structure, f, indent=2)
                                            
                                            # Add structure file path to the element data
                                            element_with_type['structure_path'] = structure_filename
                                            
                                            # Create a subdirectory for table cells
                                            table_cells_dir = image_filename.replace('.jpg', '_cells')
                                            os.makedirs(table_cells_dir, exist_ok=True)
                                            
                                            # Extract cell images from table structure
                                            if 'data' in table_structure and table_structure['data']:
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
                                                                base_name = os.path.basename(image_filename).replace('.jpg', '')
                                                                cell_image_filename = os.path.join(table_cells_dir, f"{base_name}_cell_{cell_idx+1}.jpg")
                                                                cell_image.save(cell_image_filename, "JPEG", quality=90)
                                                                
                                                                # Add cell image to OCR tasks to process later in batches
                                                                table_cell_ocr_tasks.append(cell_image_filename)
                                            
                                            # Remove temporary image used for API call
                                            os.remove(temp_table_path)
                                        except Exception as e:
                                            print(f"Error extracting table structure for {image_filename}: {str(e)}")
                                    
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
                
                # Print the result structure for debugging
                print(f"API response structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")
                
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
                                        
                                        # If this is a table, also extract table structure
                                        if content_type == 'table':
                                            # Save the cropped image temporarily for API call
                                            temp_table_path = image_filename.replace('.jpg', '_for_api.jpg')
                                            cropped_image.save(temp_table_path, "JPEG", quality=80)
                                            
                                            # Extract table structure using the cropped image
                                            try:
                                                if timing:
                                                    start_time = time.time()
                                                table_structure = extract_table_structure(temp_table_path, api_key)
                                                if timing:
                                                    table_structure_time += time.time() - start_time
                                                
                                                # Save the table structure as a JSON file
                                                structure_filename = image_filename.replace('.jpg', '_structure.json')
                                                with open(structure_filename, 'w') as f:
                                                    json.dump(table_structure, f, indent=2)
                                                
                                                # Add structure file path to the element data
                                                element_with_type['structure_path'] = structure_filename
                                                
                                                # Create a subdirectory for table cells
                                                table_cells_dir = image_filename.replace('.jpg', '_cells')
                                                os.makedirs(table_cells_dir, exist_ok=True)
                                                
                                                # Extract cell images from table structure
                                                if 'data' in table_structure and table_structure['data']:
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
                                                                    base_name = os.path.basename(image_filename).replace('.jpg', '')
                                                                    cell_image_filename = os.path.join(table_cells_dir, f"{base_name}_cell_{cell_idx+1}.jpg")
                                                                    cell_image.save(cell_image_filename, "JPEG", quality=90)
                                                                    
                                                                    # Add cell image to OCR tasks to process later in batches
                                                                    table_cell_ocr_tasks.append(cell_image_filename)
                                                
                                                # Remove temporary image used for API call
                                                os.remove(temp_table_path)
                                            except Exception as e:
                                                print(f"Error extracting table structure for {image_filename}: {str(e)}")
                                        
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
    
    # Now process all the OCR tasks in batches
    all_ocr_tasks = []
    all_ocr_tasks.extend(table_cell_ocr_tasks)
    all_ocr_tasks.extend(chart_element_ocr_tasks)
    if ocr_titles:
        all_ocr_tasks.extend(title_ocr_tasks)

    # Process OCR tasks in batches
    if all_ocr_tasks:
        print(f"Processing {len(all_ocr_tasks)} OCR tasks in batches...")
        for i in range(0, len(all_ocr_tasks), batch_size):
            ocr_batch = all_ocr_tasks[i:i + batch_size]
            print(f"Processing OCR batch {i//batch_size + 1}/{(len(all_ocr_tasks)-1)//batch_size + 1} ({len(ocr_batch)} images)")
            print(f"  Images in OCR batch: {ocr_batch}")
            
            try:
                if timing:
                    start_time = time.time()
                ocr_batch_results = extract_ocr_text_batch(ocr_batch, api_key, batch_size)
                if timing:
                    ocr_time += time.time() - start_time
                
                # Process the OCR batch results
                if 'data' in ocr_batch_results[0]:
                    batch_ocr_data = ocr_batch_results[0]['data']
                    for j, img_path in enumerate(ocr_batch):
                        if j < len(batch_ocr_data):
                            # Create individual result for this specific image
                            ocr_result = {"data": [batch_ocr_data[j]]}
                        else:
                            # Fallback: process individually if batch results don't match
                            print(f"Batch result mismatch for {img_path}, processing individually")
                            ocr_result = extract_ocr_text(img_path, api_key)
                        
                        # Save the OCR result as a JSON file
                        ocr_filename = img_path.replace('.jpg', '_ocr.json')
                        with open(ocr_filename, 'w') as f:
                            json.dump(ocr_result, f, indent=2)
            except Exception as e:
                print(f"Error processing OCR batch: {str(e)}")
                # Fall back to individual processing if batch processing fails
                for img_path in ocr_batch:
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
    if timing:
        total_time = pdf_extraction_time + page_elements_time + table_structure_time + chart_structure_time + ocr_time
        print(f"Timing Summary:")
        print(f"  PDF Extraction: {pdf_extraction_time:.2f}s")
        print(f"  Page Elements Inference: {page_elements_time:.2f}s")
        print(f"  Table Structure: {table_structure_time:.2f}s")
        print(f"  Chart Structure: {chart_structure_time:.2f}s")
        print(f"  OCR: {ocr_time:.2f}s")
        print(f"  Total: {total_time:.2f}s")


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
    print(f"Total Inference Requests: {counts['total_inference_requests']}")
    print()
    
    print("Content Type Breakdown:")
    for content_type, stats in counts['content_type_breakdown'].items():
        print(f"  {content_type}: {stats['total_elements']} elements, {stats['inference_requests']} inference requests")
    print()
    
    print("Per-Page Breakdown:")
    for page_name, page_stats in counts['pages'].items():
        print(f"  {page_name}:")
        for content_type, count in page_stats['content_types'].items():
            print(f"    {content_type}: {count} elements")
        print(f"    Total inference requests: {page_stats['inference_requests']}")


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
                            markdown_content.append(f"> {text.strip()}")
                        
                        # Now format cell data as markdown table if there's cell data
                        if cell_texts:
                            # Sort cells by their source order (cell_1, cell_2, etc.)
                            sorted_cells = sorted(cell_texts, key=lambda x: int(x[0].replace('cell_', '')) if x[0].startswith('cell_') else 0)
                            
                            # Get the table structure to properly format the table
                            table_structure_path = element.get('table_structure_path')
                            if table_structure_path and os.path.exists(table_structure_path):
                                import json
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
                                                        
                                                        markdown_content.append(header_row)
                                                        markdown_content.append(separator_row)
                                                        
                                                        # Add remaining rows
                                                        for row in rows[1:]:
                                                            row_data = [cell['text'].strip() if cell['text'].strip() else " " for cell in row]
                                                            row_str = "| " + " | ".join(row_data) + " |"
                                                            markdown_content.append(row_str)
                                                else:
                                                    # Fallback: if we cannot determine rows, just put all cells in one row
                                                    all_texts = [cell['text'].strip() if cell['text'].strip() else " " for cell in cell_data_with_coords]
                                                    if all_texts:
                                                        header_row = "| " + " | ".join(all_texts) + " |"
                                                        separator_row = "|" + "|".join([" --- " for _ in all_texts]) + "|"
                                                        markdown_content.append(header_row)
                                                        markdown_content.append(separator_row)
                            else:
                                # If no structure file, just display as list
                                for source, text in sorted_cells:
                                    if text.strip():
                                        markdown_content.append(f"- {text.strip()}")
                    else:
                        # Handle non-table content as before
                        for content in content_texts:
                            text = content.get('text', '')
                            if text.strip():
                                markdown_content.append(f"- {text.strip()}")
                
                # Add bounding box info if available
                bounding_box = element.get('bounding_box', {})
                if bounding_box:
                    markdown_content.append(f"> Bounding Box: ({bounding_box.get('x_min')}, {bounding_box.get('y_min')}) to ({bounding_box.get('x_max')}, {bounding_box.get('y_max')})")
                
                # Add image path if available (only sub-image paths, not original page images)  
                sub_image_path = element.get('sub_image_path')
                if sub_image_path:
                    # Try to make the path relative to extracts directory for proper linking
                    rel_path = os.path.relpath(sub_image_path, extract_dir) if extract_dir else os.path.basename(sub_image_path)
                    markdown_content.append(f"> Image: `{rel_path}`")
                
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
        if section.startswith('# ') or section.startswith('## Document Overview'):
            continue
        filtered_sections.append((i, section))
    
    results = []
    
    # Batch process embeddings - API typically allows up to 2048 tokens per request and multiple inputs
    batch_size = 20  # Reasonable batch size to stay within API limits
    for batch_start in range(0, len(filtered_sections), batch_size):
        batch = filtered_sections[batch_start:batch_start + batch_size]
        batch_indices, batch_sections = zip(*batch)
        
        try:
            # Generate embeddings for the batch
            response = client.embeddings.create(
                input=list(batch_sections),
                model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                encoding_format="float",
                extra_body={"modality": ["text"], "input_type": "query", "truncate": "NONE"}
            )
            
            # Process each embedding in the batch response
            for i, (page_idx, section) in enumerate(batch):
                embedding = response.data[i].embedding
                
                results.append({
                    'page_index': page_idx,
                    'content': section,
                    'embedding': embedding
                })
                
                print(f"Generated embedding for page {page_idx} (length: {len(section)} chars)")
                
        except Exception as e:
            print(f"Error generating embedding batch starting at page {batch[0][0]}: {str(e)}")
            # Fallback: process individually if batch fails
            for page_idx, section in batch:
                try:
                    response = client.embeddings.create(
                        input=[section],
                        model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
                        encoding_format="float",
                        extra_body={"modality": ["text"], "input_type": "query", "truncate": "NONE"}
                    )
                    
                    embedding = response.data[0].embedding
                    
                    results.append({
                        'page_index': page_idx,
                        'content': section,
                        'embedding': embedding
                    })
                    
                    print(f"Generated embedding for page {page_idx} (length: {len(section)} chars) using fallback")
                    
                except Exception as e_single:
                    print(f"Error generating embedding for page {page_idx}: {str(e_single)}")
                    results.append({
                        'page_index': page_idx,
                        'content': section,
                        'embedding': None,
                        'error': str(e_single)
                    })
    
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
    
