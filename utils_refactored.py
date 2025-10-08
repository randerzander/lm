import os
import json
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


def _process_table_element(element, idx, original_image, content_type_dir, image_filename, api_key):
    """Process a table element, extract structure, cells and OCR."""
    # Calculate pixel coordinates from normalized coordinates
    img_width, img_height = original_image.size
    x_min = int(element['x_min'] * img_width)
    y_min = int(element['y_min'] * img_height)
    x_max = int(element['x_max'] * img_width)
    y_max = int(element['y_max'] * img_height)
    
    # Crop the sub-image based on bounding box
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    
    # Save the cropped image in the content type directory
    image_filename = os.path.join(content_type_dir, f"{os.path.basename(image_filename).split('.')[0]}_element_{idx+1}_table.jpg")
    cropped_image.save(image_filename, "JPEG", quality=90)
    
    # Save the cropped image temporarily for API call
    temp_table_path = image_filename.replace('.jpg', '_for_api.jpg')
    cropped_image.save(temp_table_path, "JPEG", quality=80)
    
    # Extract table structure using the cropped image
    try:
        table_structure = extract_table_structure(temp_table_path, api_key)
        
        # Save the table structure as a JSON file
        structure_filename = image_filename.replace('.jpg', '_structure.json')
        with open(structure_filename, 'w') as f:
            json.dump(table_structure, f, indent=2)
        
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
                                cell_ocr_result = extract_ocr_text(cell_image_filename, api_key)
                                
                                # Save the OCR result as a JSON file
                                ocr_filename = cell_image_filename.replace('.jpg', '_ocr.json')
                                with open(ocr_filename, 'w') as f:
                                    json.dump(cell_ocr_result, f, indent=2)
                            except Exception as e:
                                print(f"Error performing OCR on cell {cell_image_filename}: {str(e)}")
        
        # Remove temporary image used for API call
        os.remove(temp_table_path)
        
        # Return element with updated info
        element_with_type = element.copy()
        element_with_type['type'] = 'table'
        element_with_type['sub_image_path'] = image_filename
        element_with_type['structure_path'] = structure_filename
    except Exception as e:
        print(f"Error extracting table structure for {image_filename}: {str(e)}")
        # Return basic element info even if table processing failed
        element_with_type = element.copy()
        element_with_type['type'] = 'table'
        element_with_type['sub_image_path'] = image_filename
    
    return element_with_type


def _process_chart_element(element, idx, original_image, content_type_dir, image_filename, api_key):
    """Process a chart element, extract graphic elements and OCR."""
    # Calculate pixel coordinates from normalized coordinates
    img_width, img_height = original_image.size
    x_min = int(element['x_min'] * img_width)
    y_min = int(element['y_min'] * img_height)
    x_max = int(element['x_max'] * img_width)
    y_max = int(element['y_max'] * img_height)
    
    # Crop the sub-image based on bounding box
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    
    # Save the cropped image in the content type directory
    image_filename = os.path.join(content_type_dir, f"{os.path.basename(image_filename).split('.')[0]}_element_{idx+1}_chart.jpg")
    cropped_image.save(image_filename, "JPEG", quality=90)
    
    # Save the cropped image temporarily for API call
    temp_chart_path = image_filename.replace('.jpg', '_for_api.jpg')
    cropped_image.save(temp_chart_path, "JPEG", quality=80)
    
    # Extract graphic elements from the chart
    try:
        graphic_elements = extract_graphic_elements(temp_chart_path, api_key)
        
        # Save the graphic elements as a JSON file
        elements_filename = image_filename.replace('.jpg', '_elements.json')
        with open(elements_filename, 'w') as f:
            json.dump(graphic_elements, f, indent=2)
        
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
                                            elem_ocr_result = extract_ocr_text(elem_image_filename, api_key)
                                            
                                            # Save the OCR result as a JSON file
                                            ocr_filename = elem_image_filename.replace('.jpg', '_ocr.json')
                                            with open(ocr_filename, 'w') as f:
                                                json.dump(elem_ocr_result, f, indent=2)
                                        except Exception as e:
                                            print(f"Error performing OCR on element {elem_image_filename}: {str(e)}")
        
        # Remove temporary image used for API call
        os.remove(temp_chart_path)
        
        # Return element with updated info
        element_with_type = element.copy()
        element_with_type['type'] = 'chart'
        element_with_type['sub_image_path'] = image_filename
        element_with_type['elements_path'] = elements_filename
    except Exception as e:
        print(f"Error extracting graphic elements for {image_filename}: {str(e)}")
        # Return basic element info even if chart processing failed
        element_with_type = element.copy()
        element_with_type['type'] = 'chart'
        element_with_type['sub_image_path'] = image_filename
    
    return element_with_type


def _process_title_element(element, idx, original_image, content_type_dir, image_filename):
    """Process a title element (no additional processing needed)."""
    # Calculate pixel coordinates from normalized coordinates
    img_width, img_height = original_image.size
    x_min = int(element['x_min'] * img_width)
    y_min = int(element['y_min'] * img_height)
    x_max = int(element['x_max'] * img_width)
    y_max = int(element['y_max'] * img_height)
    
    # Crop the sub-image based on bounding box
    cropped_image = original_image.crop((x_min, y_min, x_max, y_max))
    
    # Save the cropped image in the content type directory
    title_image_filename = os.path.join(content_type_dir, f"{os.path.basename(image_filename).split('.')[0]}_element_{idx+1}_title.jpg")
    cropped_image.save(title_image_filename, "JPEG", quality=90)
    
    # Return element with updated info
    element_with_type = element.copy()
    element_with_type['type'] = 'title'
    element_with_type['sub_image_path'] = title_image_filename
    
    return element_with_type


def process_page_images(pages_dir="pages", output_dir="page_elements"):
    """
    Process all page images in the specified directory, extract content elements,
    and save them in subdirectories organized by content type in JSONL format.
    
    Args:
        pages_dir (str): Directory containing page images
        output_dir (str): Output directory for extracted elements
    """
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
            # Extract bounding boxes
            result = extract_bounding_boxes(image_path, api_key)
            
            # Print the result structure for debugging
            print(f"API response structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")
            
            # Process the bounding box data according to the actual API response format
            if 'data' in result and result['data']:
                for page_data in result['data']:  # Each page's data
                    if 'bounding_boxes' in page_data:
                        bounding_boxes = page_data['bounding_boxes']
                        
                        # Open the original image to crop sub-images
                        original_image = Image.open(image_path)
                        
                        # Process each content type (table, chart, title, etc.)
                        for content_type, elements in bounding_boxes.items():
                            if elements:  # If there are elements of this type
                                content_type_dir = os.path.join(output_dir, content_type)
                                os.makedirs(content_type_dir, exist_ok=True)
                                
                                # Create JSONL file for this content type
                                jsonl_filename = os.path.join(content_type_dir, f"{os.path.basename(image_path).split('.')[0]}_elements.jsonl")
                                
                                # Process each element and save sub-image
                                for idx, element in enumerate(elements):
                                    if content_type == 'table':
                                        element_with_type = _process_table_element(
                                            element, idx, original_image, content_type_dir, image_path, api_key
                                        )
                                    elif content_type == 'chart':
                                        element_with_type = _process_chart_element(
                                            element, idx, original_image, content_type_dir, image_path, api_key
                                        )
                                    elif content_type == 'title':
                                        element_with_type = _process_title_element(
                                            element, idx, original_image, content_type_dir, image_path
                                        )
                                    else:
                                        # For other content types, just crop the image
                                        element_with_type = _process_title_element(
                                            element, idx, original_image, content_type_dir, image_path
                                        )
                                    
                                    # Add image path to element data
                                    element_with_type['image_path'] = image_path
                                    
                                    with open(jsonl_filename, 'a') as f:
                                        f.write(json.dumps(element_with_type) + '\n')
                                        
                        # Close the original image to free memory
                        original_image.close()
            else:
                print(f"No data found in response for {image_path}")
                print(f"Full response: {result}")
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")


if __name__ == "__main__":
    # Process all page images
    process_page_images()
    print("Page element extraction completed!")