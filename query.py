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

    Args:
        db_path (str): Path to the LanceDB directory
        query_text (str): Query text to search for
        source_document (str): Optional source document filter
        limit (int): Number of results to return (default: 5)

    Returns:
        list: List of retrieved documents with content and metadata
    """
    try:
        # Connect to the database
        db = lancedb.connect(db_path)
        
        # Find the table names - they follow the pattern {source_fn}_pages
        table_names = db.table_names()
        
        if not table_names:
            print(f"No tables found in {db_path}")
            return []
        
        # If a specific source document is provided, look for its table
        if source_document:
            table_name = f"{source_document}_pages"
            if table_name not in table_names:
                print(f"Table {table_name} not found. Available tables: {table_names}")
                return []
        else:
            # If no specific document is specified, use the first available table
            # In practice, we'll search across all tables or use the most recent
            table_name = table_names[0]  # Use the first table as default
        
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
        results = table.search(query_embedding).limit(limit).to_list()
        
        return results
        
    except Exception as e:
        print(f"Error querying LanceDB: {str(e)}")
        return []


def query_with_llm(context, question):
    """
    Query the LLM with provided context to answer a question.

    Args:
        context (str): Retrieved context from LanceDB
        question (str): Question to answer

    Returns:
        str: Generated answer from the LLM
    """
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("NVIDIA_API_KEY")
    )
    
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
            
    return answer


def main():
    parser = argparse.ArgumentParser(description="Query documents using LanceDB and LLM")
    parser.add_argument("query", nargs='?', help="Question to ask about the documents")
    parser.add_argument("--db-path", help="Path to the LanceDB directory (default: ./extracts/{source_document}/lancedb)")
    parser.add_argument("--source-document", required=True, help="Specific source document to search in (required)")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to retrieve (default: 5)")
    
    args = parser.parse_args()
    
    # Set default db path based on source document if not provided
    if not args.db_path:
        args.db_path = f"./extracts/{args.source_document}/lancedb"
    
    # Check if query is provided either as argument or to be read from stdin
    if not args.query:
        # Try to read from stdin if no query provided
        import sys
        if not sys.stdin.isatty():
            args.query = sys.stdin.read().strip()
        else:
            print("Error: No query provided. Please provide a query as an argument or pipe it to the script.")
            print("\nUsage examples:")
            print("  python query.py 'What is the capital of France?'")
            print("  python query.py --db-path /path/to/db 'Your question here'")
            print("  echo 'Your question' | python query.py")
            return 1
    
    if not os.path.exists(args.db_path):
        print(f"Error: Database path does not exist: {args.db_path}")
        
        # Suggest available documents if the extracts directory exists
        extracts_dir = "./extracts"
        if os.path.exists(extracts_dir):
            import os
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
    results = query_lancedb(args.db_path, args.query, args.source_document, args.limit)
    
    if not results:
        print("No relevant content found in the database.")
        return 1
    
    # Combine the retrieved content for context
    context = "\n\n".join([result['content'] for result in results if 'content' in result])
    
    print(f"\nFound {len(results)} relevant results. Generating answer...")
    print("\nAnswer:\n")
    
    # Get answer from LLM with the context
    query_with_llm(context, args.query)
    
    print("\n")  # Add a final newline


if __name__ == "__main__":
    main()