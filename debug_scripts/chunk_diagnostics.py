#!/usr/bin/env python3
"""
Chunk Diagnostic Script

This script analyzes your markdown files and shows how many chunks should be created.
"""

import os
import tempfile
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from healraglib import StorageManager
from healraglib.search_index_manager import SearchIndexManager

def analyze_chunking():
    """Analyze how many chunks should be created from your markdown files."""
    
    print("🔍 Chunk Analysis Diagnostic")
    print("=" * 50)
    
    # Configuration
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    container_name = os.getenv("AZURE_CONTAINER_NAME", "healrag-documents")
    chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
    chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
    
    if not connection_string:
        print("❌ AZURE_STORAGE_CONNECTION_STRING not set")
        return
    
    try:
        # Initialize storage manager
        storage_manager = StorageManager(connection_string, container_name)
        
        if not storage_manager.verify_connection():
            print("❌ Failed to connect to Azure Blob Storage")
            return
        
        # Get markdown files
        all_files = storage_manager.get_file_list(as_json=False)
        md_files = [f for f in all_files if f.startswith("md_files/") and f.endswith('.md')]
        
        print(f"📁 Found {len(md_files)} markdown files in md_files/ folder")
        
        if len(md_files) == 0:
            print("❌ No markdown files found. Run the content extraction pipeline first.")
            return
        
        # Initialize search index manager for chunking analysis
        search_manager = SearchIndexManager(
            storage_manager=storage_manager,
            azure_openai_endpoint="dummy",  # Not needed for chunking analysis
            azure_openai_key="dummy",
            azure_openai_deployment="dummy",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        print(f"📊 Chunk Settings:")
        print(f"   Chunk size: {chunk_size} characters")
        print(f"   Chunk overlap: {chunk_overlap} characters")
        print()
        
        total_chunks_expected = 0
        total_content_length = 0
        
        print("📄 Per-file analysis:")
        
        for i, md_file in enumerate(md_files, 1):
            try:
                print(f"\n{i}. {Path(md_file).name}")
                
                # Download and read the markdown file
                with tempfile.NamedTemporaryFile(suffix='.md', delete=False) as temp_file:
                    temp_file_path = temp_file.name
                
                try:
                    if not storage_manager.download_file(md_file, temp_file_path):
                        print(f"   ❌ Failed to download {md_file}")
                        continue
                    
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    content_length = len(content)
                    total_content_length += content_length
                    
                    # Analyze chunking
                    chunks = search_manager.chunk_markdown_content(content, md_file)
                    chunks_count = len(chunks)
                    total_chunks_expected += chunks_count
                    
                    print(f"   📏 Content length: {content_length:,} characters")
                    print(f"   🔢 Expected chunks: {chunks_count}")
                    print(f"   📊 Avg chunk size: {content_length // chunks_count if chunks_count > 0 else 0:,} chars")
                    
                    # Show first few chunk details
                    if chunks_count > 0:
                        print(f"   📋 Chunk breakdown:")
                        for j, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                            print(f"      {j+1}. ID: {chunk['id']}")
                            print(f"         Size: {chunk['chunk_size']} chars")
                            print(f"         Section: {chunk['section']}")
                            print(f"         Preview: {chunk['content'][:100]}...")
                        
                        if chunks_count > 3:
                            print(f"      ... and {chunks_count - 3} more chunks")
                    
                finally:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                        
            except Exception as e:
                print(f"   ❌ Error processing {md_file}: {e}")
        
        print(f"\n📊 SUMMARY:")
        print(f"   Total markdown files: {len(md_files)}")
        print(f"   Total content: {total_content_length:,} characters")
        print(f"   Expected total chunks: {total_chunks_expected}")
        print(f"   Average chunks per file: {total_chunks_expected / len(md_files) if md_files else 0:.1f}")
        
        print(f"\n🎯 WHAT THIS MEANS:")
        if total_chunks_expected > 9:
            print(f"   ❌ Expected {total_chunks_expected} chunks but only 9 in your index!")
            print(f"   💡 This indicates a problem with:")
            print(f"      - Chunk processing/creation")
            print(f"      - Embedding generation failures")
            print(f"      - Upload failures to Azure Search")
            print(f"      - Batch processing errors")
        elif total_chunks_expected == 9:
            print(f"   ✅ Expected 9 chunks and got 9 - this is correct!")
            print(f"   💡 Each file created exactly 1 chunk on average")
        else:
            print(f"   🤔 Expected {total_chunks_expected} but got 9 in index")
        
        print(f"\n💡 NEXT STEPS:")
        print(f"   1. If chunks expected > 9: Re-run the search indexing with verbose logging")
        print(f"   2. Check for embedding generation errors in the pipeline output")
        print(f"   3. Verify Azure Search upload success rates")
        print(f"   4. Consider reducing chunk_size if files are very small")
        
        # Check Azure Search index
        azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        azure_search_key = os.getenv('AZURE_SEARCH_KEY')
        azure_search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'healrag-index')
        
        if azure_search_endpoint and azure_search_key:
            print(f"\n🔍 Azure Search Index Check:")
            try:
                from azure.search.documents import SearchClient
                from azure.core.credentials import AzureKeyCredential
                
                search_client = SearchClient(
                    endpoint=azure_search_endpoint,
                    index_name=azure_search_index_name,
                    credential=AzureKeyCredential(azure_search_key)
                )
                
                # Get actual document count
                search_results = search_client.search(
                    search_text="*",
                    include_total_count=True,
                    top=0
                )
                actual_count = search_results.get_count()
                print(f"   📊 Actual documents in '{azure_search_index_name}': {actual_count}")
                
                if actual_count != total_chunks_expected:
                    print(f"   ⚠️  Mismatch: Expected {total_chunks_expected}, Found {actual_count}")
                else:
                    print(f"   ✅ Perfect match!")
                
            except Exception as search_error:
                print(f"   ❌ Could not check Azure Search: {search_error}")
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    analyze_chunking()