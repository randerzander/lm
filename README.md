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

### 3. Query the Results

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
- **Semantic Search**: Query documents using natural language
- **LanceDB Storage**: Stores content in a queryable vector database
- **Markdown Generation**: Creates structured markdown representations

## Requirements

- Python 3.8+
- NVIDIA API key
- PDF documents to process