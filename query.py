#!/usr/bin/env python3
"""
Query script for retrieving context from LanceDB and answering questions using an LLM.
"""

import argparse
import os
import lancedb
from openai import OpenAI


def query_lancedb(db_path, query_text, source_document=None, limit=5):
    """
    Query the LanceDB collection to retrieve relevant content based on the query text.
    Now searches across all documents in a single collection.

    Args:
        db_path (str): Path to the LanceDB directory
        query_text (str): Query text to search for
        source_document (str): Optional source document filter (deprecated - now searches all documents)
        limit (int): Number of results to return (default: 5)

    Returns:
        list: List of retrieved documents with content and metadata
    """
    try:
        # Connect to the database
        db = lancedb.connect(db_path)
        
        # The table name is now fixed as "all_documents"
        table_name = "all_documents"
        
        # Check if the table exists
        table_names = db.table_names()
        if table_name not in table_names:
            print(f"Table '{table_name}' not found. Available tables: {table_names}")
            return []
        
        # Get the table
        table = db.open_table(table_name)
        
        # Perform vector search using the query text as embedding
        # First, we need to create an embedding for the query
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("NVIDIA_API_KEY")
        )
        
        response = client.embeddings.create(
            input=[query_text],
            model="nvidia/llama-3.2-nemoretriever-1b-vlm-embed-v1",
            encoding_format="float",
            extra_body={"modality": ["text"], "input_type": "query", "truncate": "NONE"}
        )
        
        query_embedding = response.data[0].embedding
        
        # Query the table using vector search
        # If source_document is provided, filter by that document
        if source_document:
            results = table.search(query_embedding).where(f"source_document = '{source_document}'").limit(limit).to_list()
        else:
            # Search across all documents
            results = table.search(query_embedding).limit(limit).to_list()
        
        return results
        
    except Exception as e:
        print(f"Error querying LanceDB: {str(e)}")
        return []


def query_with_llm(context, question, result_metadata):
    """
    Query the LLM with provided context to answer a question.

    Args:
        context (str): Retrieved context from LanceDB
        question (str): Question to answer
        result_metadata (dict): Metadata for the result including source_document, page_index, and other fields

    Returns:
        str: Generated answer from the LLM
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
    # Extract metadata
    source_document = result_metadata.get('source_document', 'unknown')
    page_number = result_metadata.get('page_number', 'unknown')  # Fix: Use correct field name from LanceDB schema
    # Clean up page number formatting - remove leading zeros
    if page_number != 'unknown' and isinstance(page_number, str) and page_number.isdigit():
        page_index = str(int(page_number))  # Convert "001" to "1"
    else:
        page_index = page_number
    
    # Check content type - based on the structure, we can infer content type from the context content
    content_type = "text"
    # Check if the content contains table-like structures, chart descriptions, or other elements
    context_lower = context.lower()
    if any(keyword in context_lower for keyword in ['table', '|', 'column', 'row', 'cell', 'headers']):
        content_type = "table"
    elif any(keyword in context_lower for keyword in ['chart', 'graph', 'plot', 'axis', 'legend', 'data point', 'bar', 'pie']):
        content_type = "chart"
    elif any(keyword in context_lower for keyword in ['title', 'heading', 'subtitle']):
        content_type = "title"
    elif any(keyword in context_lower for keyword in ['image', 'picture', 'figure', 'diagram']):
        content_type = "image"
    
    # Prepare the prompt with context and question
    system_message = f"""
    You are a helpful assistant that answers questions based on provided context.
    
    Context: {context}
    
    Answer the following question using only the information provided in the context.
    If the answer is not available in the context, say "I don't have enough information to answer that question."
    """
    
    completion = client.chat.completions.create(
        model="nvidia/nvidia-nemotron-nano-9b-v2",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ],
        temperature=0.6,
        top_p=0.95,
        max_tokens=2048,
        frequency_penalty=0,
        presence_penalty=0,
        stream=True,
        extra_body={
            "min_thinking_tokens": 1024,
            "max_thinking_tokens": 2048
        }
    )

    answer = ""
    for chunk in completion:
        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="", flush=True)
        if chunk.choices[0].delta.content is not None:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            answer += content
            
    # Print citation information after the answer
    print(f"\n\n[Source: Document '{source_document}', Page {page_index}, Content Type: {content_type}]")
            
    return answer


def main():
    parser = argparse.ArgumentParser(description="Query documents using LanceDB and LLM")
    parser.add_argument("query", nargs='?', help="Question to ask about the documents")
    parser.add_argument("--db-path", help="Path to the LanceDB directory (default: ./lancedb)")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve (default: 5)")
    
    args = parser.parse_args()
    
    # Set default db path if not provided
    if not args.db_path:
        # Default to a common lancedb path that can contain all documents
        # If multiple document directories exist, we'll look for the first one with a lancedb directory
        import glob
        db_paths = glob.glob("./extracts/*/lancedb")
        if db_paths:
            # Use the first available lancedb directory
            args.db_path = db_paths[0]
        else:
            # Default fallback
            args.db_path = "./lancedb"
    
    # Check if query is provided either as argument or to be read from stdin
    if not args.query:
        # Try to read from stdin if no query provided
        import sys
        if not sys.stdin.isatty():
            args.query = sys.stdin.read().strip()
        else:
            print("Error: No query provided. Please provide a query as an argument or pipe it to the script.")
            print("\nUsage examples:")
            print("  python query.py 'What is the capital of France?'  # Search across all documents")
            print("  python query.py --source-document doc1 'Your question here'  # Search specific document")
            print("  python query.py --db-path /path/to/db 'Your question here'")
            print("  echo 'Your question' | python query.py")
            return 1
    
    if not os.path.exists(args.db_path):
        print(f"Error: Database path does not exist: {args.db_path}")
        
        # Suggest available documents if the extracts directory exists
        extracts_dir = "./extracts"
        if os.path.exists(extracts_dir):
            available_docs = [d for d in os.listdir(extracts_dir) 
                             if os.path.isdir(os.path.join(extracts_dir, d)) 
                             and os.path.exists(os.path.join(extracts_dir, d, "lancedb"))]
            if available_docs:
                print(f"\nAvailable documents with LanceDB databases:")
                for doc in available_docs:
                    print(f"  --source-document {doc}")
                print(f"\nExample: python query.py --source-document {available_docs[0]} '{args.query}'")
        
        return 1
    
    # Query the database to get relevant context
    print("Searching for relevant context...")
    results = query_lancedb(args.db_path, args.query, None, args.limit)
    
    if not results:
        print("No relevant content found in the database.")
        return 1
    
    print(f"\nFound {len(results)} relevant results. Using top result for answer generation...")
    
    # Print the retrieved chunks before generating the answer
    #print("\nRetrieved content chunks:")
    #print("="*50)
    for i, result in enumerate(results, 1):
        source_document = result.get('source_document', 'unknown')
        page_number = result.get('page_number', 'unknown')  # Fix: Use correct field name from LanceDB schema
        # Clean up page number formatting - remove leading zeros
        if page_number != 'unknown' and isinstance(page_number, str) and page_number.isdigit():
            page_index = str(int(page_number))  # Convert "001" to "1"
        else:
            page_index = page_number
        content_length = result.get('page_content_length', len(result.get('content', '')))
        
        #print(f"\nChunk {i} (Document: {source_document}, Page: {page_index}, Length: {content_length} chars):")
        #print("-" * 50)
        #print(result['content'])
        #print("-" * 50)
    
    print("\nAnswer:\n")
    
    # Use only the top result for context
    top_result = results[0]
    context = top_result['content']
    
    # Get answer from LLM with the top result as context
    query_with_llm(context, args.query, top_result)
    
    print("\n")  # Add a final newline


if __name__ == "__main__":
    main()
