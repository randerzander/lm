# LM - PDF Content Extraction with AI-Powered Element Detection

LM is a powerful tool for extracting structured content from PDF documents using NVIDIA AI APIs. It processes PDFs to identify and extract various content elements like tables, charts, and titles, along with their structure and OCR text.

## Features

- **Multi-modal PDF Processing**: Extracts both visual and textual content from PDF documents
- **AI-Powered Element Detection**: Uses NVIDIA AI APIs to identify content elements (tables, charts, titles)
- **Element Structure Extraction**: Gets detailed structure information for tables and charts
- **OCR Integration**: Performs OCR on detected elements to extract text content
- **Configurable Multiprocessing**: Process PDF pages in parallel with adjustable batching strategies
- **Structured Output**: Provides content in organized JSON format with metadata and file references

## Prerequisites

- Python 3.8+
- NVIDIA API key (set as environment variable `NVIDIA_API_KEY`)
- PDF documents to process

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:randerzander/lm.git
   cd lm
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your NVIDIA API key:
   ```bash
   export NVIDIA_API_KEY="your-api-key-here"
   ```

## Usage

### Basic Usage

Process a PDF document with default settings:
```bash
python process_pdf.py
```

### Advanced Options

The tool supports various command-line options for processing:

#### Single PDF Processing

Process a single PDF file with default settings:
```bash
python process_pdf.py
```

Process a specific PDF file:
```bash
python process_pdf.py /path/to/your/document.pdf
```

Process with custom pages per process and max processes:
```bash
# python process_pdf.py [pages_per_process] [max_processes] [pdf_path]
python process_pdf.py 2 4 /path/to/your/document.pdf
```

#### Directory Processing (NEW!)

Process all PDFs in a directory:
```bash
python process_pdf.py /path/to/directory/
```

Process all PDFs in a directory with custom settings:
```bash
# python process_pdf.py [pages_per_process] [max_processes] [directory_path]
python process_pdf.py 2 4 /path/to/directory/
```

#### Multiprocessing Control

Control the multiprocessing behavior:

```bash
# Process 1 page per process (maximum parallelization)
python process_pdf.py 1

# Process 2 pages per process (balanced approach)
python process_pdf.py 2

# Process with 2 pages per process and up to 8 parallel processes
python process_pdf.py 2 8

# Process all pages in a single process (minimal overhead)
python process_pdf.py 10
```

### Directory Structure

```
lm/
├── data/                 # Input PDF documents
├── scratch/               # Temporary processing files
│   ├── texts/            # Extracted text from PDF pages
│   ├── pages/            # Rendered page images
│   └── page_elements/    # Detected content elements
├── process_pdf.py        # Main processing script
├── utils.py             # Utility functions
└── requirements.txt      # Python dependencies
```

## How It Works

### PDF Processing Pipeline

1. **PDF Extraction**: Converts PDF pages to images and extracts text
2. **Element Detection**: Uses NVIDIA AI APIs to detect content elements
3. **Structure Analysis**: Gets detailed structure for tables and charts
4. **OCR Processing**: Performs OCR on detected elements
5. **Consolidation**: Combines all extracted content into structured JSON output

### Multiprocessing Architecture

The tool implements intelligent multiprocessing for optimal performance:

- **Dynamic Process Allocation**: Automatically determines optimal number of processes based on CPU cores and document size
- **Configurable Batching**: Control how many pages are processed per process with the `pages_per_process` parameter
- **Resource Management**: Caps concurrent processes to prevent system overload

#### Processing Strategies

1. **High Parallelization** (`pages_per_process=1`):
   - Each page processed in its own process
   - Maximum CPU utilization
   - Higher overhead but maximum concurrency

2. **Balanced Processing** (`pages_per_process=2-N`):
   - Groups pages into batches
   - Reduces process creation overhead
   - Good balance of performance and efficiency

3. **Single Process** (`pages_per_process>=total_pages`):
   - All pages processed sequentially in one process
   - Minimal overhead
   - Best for small documents or limited system resources

### AI APIs Used

- **Page Elements Detection**: Identifies tables, charts, titles, and other content elements
- **Table Structure Analysis**: Extracts detailed table structure with cell boundaries
- **Chart Element Recognition**: Identifies chart components like titles, axes, legends
- **OCR**: Extracts text from detected elements

## Configuration

Key configuration options in `process_pdf.py`:

- `pages_per_process`: Number of pages to process in each multiprocessing unit (default: 1)
- `max_processes`: Maximum number of parallel processes to use (default: system CPU count)
- `ocr_titles`: Whether to perform OCR on title elements (default: False)
- `timing`: Enable detailed timing reports (default: False)

Environment variables:
- `NVIDIA_API_KEY`: Required for accessing NVIDIA AI APIs

## Output Format

The tool generates structured JSON output containing:

- **Element Metadata**: Type, bounding boxes, confidence scores
- **File References**: Paths to extracted images and OCR results
- **Text Content**: Extracted text from OCR processing
- **Structure Data**: Detailed structure information for tables and charts

## Performance Considerations

### Optimal Settings

For best performance, consider:

1. **Large Documents (50+ pages)**:
   ```bash
   python process_pdf.py 3 4
   ```
   Process 3 pages per process with up to 4 parallel processes to balance parallelization with overhead.

2. **Small Documents (<10 pages)**:
   ```bash
   python process_pdf.py 1 8
   ```
   Maximum parallelization with up to 8 processes for fastest processing.

3. **Limited System Resources**:
   ```bash
   python process_pdf.py 2 2
   ```
   Reduce concurrent processes to minimize resource usage.

4. **Many PDFs in Directory**:
   ```bash
   python process_pdf.py 1 2 /path/to/directory/
   ```
   Process with fewer parallel processes when handling multiple files to avoid system overload.

### Timing Reports

Enable timing reports to analyze performance:
```bash
python process_pdf.py 1  # with timing=True in extract() call
```

Sample timing output:
```
Timing Summary:
  PDF Extraction: 0.06s
  Page Elements Inference: 2.04s
  Table Structure: 0.97s
  Chart Structure: 1.30s
  OCR: 24.78s
  Total: 29.15s
```

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   ```
   NVIDIA_API_KEY environment variable not set
   ```
   Solution: Ensure `NVIDIA_API_KEY` is properly exported.

2. **PDF Processing Failures**:
   ```
   API request failed with status 400
   ```
   Solution: Check PDF format and ensure it meets API requirements.

3. **Memory Issues with Large Documents**:
   Solution: Reduce `pages_per_process` value or process documents in smaller batches.

### Debugging Tips

- Enable timing reports to identify bottlenecks
- Check API response structures for unexpected formats
- Monitor system resources during processing
- Use smaller test documents for development

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- Performance improvements
- New features
- Documentation enhancements

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA for providing the AI APIs used in this project
- The open-source community for various Python libraries and tools