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

    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()


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

    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"API request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()


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

    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"OCR API request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()


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

    response = requests.post(invoke_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Graphic elements API request failed with status {response.status_code}: {response.text}")
        response.raise_for_status()

def process_page_images(pages_dir="pages", output_dir="page_elements", timing=False, ocr_titles=False):
    """
    Process all page images in the specified directory, extract content elements,
    and save them in subdirectories organized by content type in JSONL format.
    
    Args:
        pages_dir (str): Directory containing page images
        output_dir (str): Output directory for extracted elements
        timing (bool): Whether to track and report timing for each stage
        ocr_titles (bool): Whether to perform OCR on title elements
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
    
    # Process all page images
    page_images = glob(os.path.join(pages_dir, "*.jpg")) + glob(os.path.join(pages_dir, "*.png"))
    
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
                                                                
                                                                # Extract OCR text from the cell image
                                                                try:
                                                                    if timing:
                                                                        start_time = time.time()
                                                                    cell_ocr_result = extract_ocr_text(cell_image_filename, api_key)
                                                                    if timing:
                                                                        ocr_time += time.time() - start_time
                                                                    
                                                                    # Save the OCR result as a JSON file
                                                                    ocr_filename = cell_image_filename.replace('.jpg', '_ocr.json')
                                                                    with open(ocr_filename, 'w') as f:
                                                                        json.dump(cell_ocr_result, f, indent=2)
                                                                except Exception as e:
                                                                    print(f"Error performing OCR on cell {cell_image_filename}: {str(e)}")
                                            
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
                                                                            
                                                                            # Extract OCR text from the element image
                                                                            try:
                                                                                if timing:
                                                                                    start_time = time.time()
                                                                                elem_ocr_result = extract_ocr_text(elem_image_filename, api_key)
                                                                                if timing:
                                                                                    ocr_time += time.time() - start_time
                                                                                
                                                                                # Save the OCR result as a JSON file
                                                                                ocr_filename = elem_image_filename.replace('.jpg', '_ocr.json')
                                                                                with open(ocr_filename, 'w') as f:
                                                                                    json.dump(elem_ocr_result, f, indent=2)
                                                                            except Exception as e:
                                                                                print(f"Error performing OCR on element {elem_image_filename}: {str(e)}")
                                            
                                            # Remove temporary image used for API call
                                            os.remove(temp_chart_path)
                                        except Exception as e:
                                            print(f"Error extracting graphic elements for {image_filename}: {str(e)}")
                                    
                                    # If this is a title, perform OCR on the cropped title image if requested
                                    elif content_type == 'title':
                                        if ocr_titles:
                                            # Perform OCR on the title image
                                            try:
                                                if timing:
                                                    start_time = time.time()
                                                title_ocr_result = extract_ocr_text(image_filename, api_key)
                                                
                                                # Save the OCR result as a JSON file
                                                ocr_filename = image_filename.replace('.jpg', '_ocr.json')
                                                with open(ocr_filename, 'w') as f:
                                                    json.dump(title_ocr_result, f, indent=2)
                                                
                                                if timing:
                                                    ocr_time += time.time() - start_time
                                            except Exception as e:
                                                print(f"Error performing OCR on title {image_filename}: {str(e)}")
                                    
                                    with open(jsonl_filename, 'a') as f:
                                        f.write(json.dumps(element_with_type) + '\n')
                                        
                        # Close the original image to free memory
                        original_image.close()
            else:
                print(f"No data found in response for {image_path}")
                print(f"Full response: {result}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
    
    # Report timing if requested
    if timing:
        total_time = page_elements_time + table_structure_time + chart_structure_time + ocr_time
        print(f"Timing Summary:")
        print(f"  PDF Extraction: 0.00s (not measured in this function)")
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
    
    # Process original page images to add to the result
    for image_path in sorted(glob(os.path.join(pages_dir, "*.jpg")) + glob(os.path.join(pages_dir, "*.png"))):
        page_name = os.path.basename(image_path).replace('.jpg', '').replace('.png', '')
        
        if page_name not in result['pages']:
            result['pages'][page_name] = {
                'original_image_path': image_path,
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
    
                            # Add to page elements
                            result["pages"][page_name]["elements"].append(element_record)
                            result["total_elements"] += 1
    
    return result

def save_extracted_content_to_json(result_obj, output_file="extracted_content.json"):
    """
    Save the extracted content result object to a JSON file.
    
    Args:
        result_obj (dict): The result object from get_all_extracted_content
        output_file (str): Output file path
    """
    with open(output_file, "w") as f:
        json.dump(result_obj, f, indent=2)
    print(f"Extracted content saved to {output_file}")
