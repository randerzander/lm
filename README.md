# LM - PDF Content Extraction with AI

A tool for extracting structured content from PDF documents using NVIDIA AI APIs. Processes PDFs to identify tables, charts, and text with semantic search capabilities.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv lm
source lm/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set your NVIDIA API key
export NVIDIA_API_KEY="your-api-key-here"
```

### 2. Process a PDF

```bash
# Process the sample document
python process_pdf.py data/multimodal_test.pdf

# Or process all PDFs in a directory
python process_pdf.py data/
```

### 3. Advanced Configuration

The tool supports configurable rate limiting for API requests:

```bash
# Process with custom rate limiting parameters
# Arguments: pdf_path pages_per_process max_processes max_concurrent_requests requests_per_minute max_workers
python process_pdf.py data/multimodal_test.pdf 1 4 5 30 4
```

Or use the Python API directly:
```python
from process_pdf import extract

# Extract with custom rate limiting
result = extract(
    pdf_path="data/multimodal_test.pdf",
    max_concurrent_requests=5,      # Max concurrent API requests (default: 10)
    requests_per_minute=30,         # Max requests per minute (default: 40)
    max_workers=4                   # Max workers for thread pools (default: system CPU count)
)
```

### 4. Query the Results

```bash
# Ask questions about the processed documents
python query.py "what animal is most likely responsible for typoes, given its activity"

# The system will analyze content and provide answers like:
# "The cat is most likely responsible for typos, as its activity involves 
# jumping onto a laptop (a device used for typing), which could result 
# in accidental key presses."
# [Source: Document 'multimodal_test', Page 1, Content Type: table]
```

## Features

- **AI-Powered Element Detection**: Identifies tables, charts, and titles in PDFs
- **OCR Integration**: Extracts text from detected elements  
- **Configurable Rate Limiting**: Control concurrent requests and requests per minute
- **Semantic Search**: Query documents using natural language
- **LanceDB Storage**: Stores content in a queryable vector database
- **Markdown Generation**: Creates structured markdown representations

## Rate Limiting Configuration

The tool includes sophisticated rate limiting with two parameters:

- `max_concurrent_requests`: Maximum number of simultaneous API requests (default: 10)
- `requests_per_minute`: Maximum number of requests allowed per minute (default: 40)
- `max_workers`: Maximum number of workers for thread pools (default: system CPU count)

This dual-layer approach prevents both concurrent overload and time-based rate limits.

## Requirements

- Python 3.8+
- NVIDIA API key
- PDF documents to process