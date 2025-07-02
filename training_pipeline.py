#!/usr/bin/env python3
"""
HEALRAG Sample Pipeline

This script demonstrates the complete HEALRAG pipeline including:
- Azure Blob Storage management
- Content extraction with MarkItDown
- Image extraction from PDFs and Word documents
- Azure OpenAI Vision analysis for image descriptions
- Comprehensive markdown generation
- RAG-ready content preparation
- Search index creation with chunking and embeddings
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

from healraglib import StorageManager
from healraglib.content_manager import ContentManager

# Set up logging - reduce verbosity
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress Azure SDK verbose logging
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.ERROR)
logging.getLogger('azure.storage.blob').setLevel(logging.ERROR)

def main():
    """Main pipeline function."""
    
    print("🚀 HEALRAG Sample Pipeline")
    print("=" * 60)
    
    # Configuration
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    container_name = os.getenv("AZURE_CONTAINER_NAME", "healrag-documents")
    
    # Azure OpenAI configuration
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    if not connection_string:
        print("❌ Please set AZURE_STORAGE_CONNECTION_STRING environment variable")
        return
    
    try:
        # Step 1: Initialize Storage Manager
        print("\n📦 Step 1: Initializing Storage Manager...")
        storage_manager = StorageManager(connection_string, container_name)
        
        if not storage_manager.verify_connection():
            print("❌ Failed to connect to Azure Blob Storage")
            return
        
        print("✅ Connected to Azure Blob Storage")
        
        # Step 2: Get container statistics
        print("\n📊 Step 2: Getting container statistics...")
        stats = storage_manager.get_container_statistics()
        print(f"   Container: {stats['container_name']}")
        print(f"   Total files: {stats['total_files']}")
        print(f"   Total size: {stats['total_size_mb']} MB")
        
        if stats['file_types']:
            print("   File types:")
            for ext, data in stats['file_types'].items():
                print(f"     {ext}: {data['count']} files ({data['total_size_mb']} MB)")
        
        # Step 3: Initialize Content Manager
        print("\n📚 Step 3: Initializing Content Manager...")
        content_manager = ContentManager(
            storage_manager=storage_manager,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            azure_openai_deployment=azure_openai_deployment
        )
        
        # Check Azure OpenAI configuration
        if all([azure_openai_endpoint, azure_openai_key, azure_openai_deployment]):
            print("✅ Azure OpenAI configured for image analysis")
        else:
            print("⚠️  Azure OpenAI not fully configured - images will be extracted without analysis")
        
        # Step 4: List all files in the container
        print("\n📄 Step 4: Analyzing available files...")
        all_files = storage_manager.get_file_list(as_json=False)
        print(f"   Found {len(all_files)} files in container")
        
        # Get source files ready for processing (excludes md_files and unsupported types)
        supported_files = content_manager.get_source_files_from_container()
        print(f"   Source files ready for processing: {len(supported_files)}")
        for i, f in enumerate(supported_files, 1):
            print(f"     {i}. {f}")
        
        # Step 5: Show image extraction capabilities
        print("\n🖼️  Step 5: Image extraction capabilities...")
        image_types = content_manager.get_image_extraction_support()
        print(f"   Image extraction supported for: {', '.join(image_types)}")
        
        # Count files with images
        image_files = [f for f in supported_files if any(f.lower().endswith(ext) for ext in image_types)]
        print(f"   Files potentially containing images: {len(image_files)}")
        
        # Step 6: Process all supported files
        print("\n🔍 Step 6: Processing files with content extraction...")
        print(f"   Processing {len(supported_files)} files...")
        print("   Starting content extraction...")
        start_time = time.time()
        results = content_manager.extract_content_from_files(supported_files, output_folder="md_files", extract_images=True)
        print(f"   Content extraction completed. Results type: {type(results)}")
        print(f"   Number of results: {len(results)}")
        for fname, res in results.items():
            print(f"     {fname}: {type(res)} - {res.get('markdown_file') if isinstance(res, dict) else res}")
        
        elapsed = time.time() - start_time
        
        # Step 7: Print summary
        print("\n📊 Step 7: Processing Results")
        print(f"   Processing time: {elapsed:.2f} seconds")
        files_processed = sum(1 for r in results.values() if isinstance(r, dict) and r.get('success'))
        total_images = sum(r.get('images_processed', 0) for r in results.values() if isinstance(r, dict))
        total_chars = sum(r.get('content_length', 0) for r in results.values() if isinstance(r, dict))
        print(f"\nPipeline Summary:")
        print(f"   Files processed: {files_processed}/{len(supported_files)}")
        print(f"   Total images extracted: {total_images}")
        print(f"   Total content generated: {total_chars:,} characters")
        print(f"   Markdown files created in: md_files/")
        print(f"   Processing time: {elapsed:.2f} seconds\n")
        print(f"\n📁 Generated markdown files:")
        for res in results.values():
            if isinstance(res, dict) and res.get('success'):
                print(f"   📄 {res['markdown_file']}")
        
        # Step 8: Initialize Search Index Manager
        print("\n🔍 Step 8: Initializing Search Index Manager...")
        from healraglib.search_index_manager import SearchIndexManager
        
        # Get environment variables for search indexing
        azure_text_embedding_model = os.getenv('AZURE_TEXT_EMBEDDING_MODEL')
        azure_openai_deployment_for_embeddings = os.getenv('AZURE_OPENAI_DEPLOYMENT')
        azure_search_endpoint = os.getenv('AZURE_SEARCH_ENDPOINT')
        azure_search_key = os.getenv('AZURE_SEARCH_KEY')
        azure_search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME', 'healrag-index')
        chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        
        # Debug: Show environment variable configuration
        print(f"   🔧 Environment Variable Check:")
        print(f"     AZURE_TEXT_EMBEDDING_MODEL: {azure_text_embedding_model}")
        print(f"     AZURE_OPENAI_DEPLOYMENT: {azure_openai_deployment_for_embeddings}")
        print(f"     AZURE_SEARCH_ENDPOINT: {azure_search_endpoint}")
        print(f"     AZURE_SEARCH_KEY: {'***' if azure_search_key else 'None'}")
        print(f"     AZURE_SEARCH_INDEX_NAME: {azure_search_index_name}")
        
        # Determine which embedding model to use
        if azure_text_embedding_model:
            embedding_model_to_use = azure_text_embedding_model
            print(f"   ✅ Using AZURE_TEXT_EMBEDDING_MODEL: {azure_text_embedding_model}")
            
            # Validate that it looks like an embedding model
            if "embedding" not in azure_text_embedding_model.lower():
                print(f"   ⚠️  Warning: Model name doesn't contain 'embedding' - this might not be an embedding model")
                
        elif azure_openai_deployment_for_embeddings:
            embedding_model_to_use = azure_openai_deployment_for_embeddings
            print(f"   ⚠️  Using AZURE_OPENAI_DEPLOYMENT: {azure_openai_deployment_for_embeddings}")
            print(f"   💡 Consider using AZURE_TEXT_EMBEDDING_MODEL for embeddings")
            
            # Check if it looks like a chat model (which won't work for embeddings)
            if "gpt" in azure_openai_deployment_for_embeddings.lower() and "embedding" not in azure_openai_deployment_for_embeddings.lower():
                print(f"   ❌ Warning: This looks like a GPT model, not an embedding model!")
                print(f"   💡 Please use AZURE_TEXT_EMBEDDING_MODEL=text-embedding-ada-002")
        else:
            embedding_model_to_use = None
            print(f"   ❌ No embedding model specified")
            print(f"   💡 Please set AZURE_TEXT_EMBEDDING_MODEL=text-embedding-ada-002")
        
        search_manager = SearchIndexManager(
            storage_manager=storage_manager,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            azure_openai_deployment=embedding_model_to_use,
            azure_search_endpoint=azure_search_endpoint,
            azure_search_key=azure_search_key,
            azure_search_index_name=azure_search_index_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        if all([azure_openai_endpoint, azure_openai_key, embedding_model_to_use]):
            print("✅ Azure OpenAI configured for embeddings")
        else:
            print("⚠️  Azure OpenAI not fully configured for embeddings")
            if not embedding_model_to_use:
                print("   💡 Please set AZURE_TEXT_EMBEDDING_MODEL=text-embedding-ada-002")
        
        if all([azure_search_endpoint, azure_search_key]):
            print("✅ Azure Cognitive Search configured")
        else:
            print("⚠️  Azure Cognitive Search not configured")
        
        print(f"   Chunk size: {chunk_size} characters")
        print(f"   Chunk overlap: {chunk_overlap} characters")
        print(f"   Search index name: {azure_search_index_name}")
        
        # Step 9: Process markdown files for search index
        print("\n📚 Step 9: Processing markdown files for search index...")
        if files_processed > 0:
            # Check if embedding model is properly configured
            if not embedding_model_to_use:
                print("   ❌ Skipping search indexing - no embedding model configured")
                print("   💡 Please set AZURE_TEXT_EMBEDDING_MODEL=text-embedding-ada-002")
                print("   📁 Markdown files are still available for manual processing")
            else:
                print("   Starting markdown processing and indexing...")
                index_start_time = time.time()
                
                index_results = search_manager.process_markdown_files("md_files")
                
                index_elapsed = time.time() - index_start_time
                
                if index_results.get('success'):
                    print("✅ Search index processing completed successfully")
                    print(f"   Files processed: {index_results['files_processed']}")
                    print(f"   Total chunks created: {index_results['total_chunks']}")
                    print(f"   Chunks with embeddings: {index_results['chunks_with_embeddings']}")
                    print(f"   Chunks without embeddings: {index_results['chunks_without_embeddings']}")
                    print(f"   Processing errors: {index_results['processing_errors']}")
                    if index_results.get('upload_errors'):
                        print(f"   ⚠️  Upload errors occurred (some chunks may not be indexed)")
                    print(f"   Indexing time: {index_elapsed:.2f} seconds")
                    
                    # Step 10: Test search functionality
                    print("\n🔎 Step 10: Testing search functionality...")
                    if index_results['chunks_with_embeddings'] > 0:
                        test_query = "cyber security policy"
                        print(f"   Testing search with query: '{test_query}'")
                        
                        search_results = search_manager.search_similar_chunks(test_query, top_k=3)
                        
                        if search_results:
                            print("   Search results:")
                            for i, result in enumerate(search_results, 1):
                                print(f"     {i}. {result['source_file']} (Score: {result['score']:.3f})")
                                print(f"        Section: {result['section']}")
                                print(f"        Content preview: {result['content'][:100]}...")
                        else:
                            print("   No search results found")
                    else:
                        print("   Skipping search test - no embeddings available")
                else:
                    print(f"❌ Search index processing failed: {index_results.get('error', 'Unknown error')}")
                    print(f"   However, markdown files were still generated successfully")
                    print(f"   You can manually upload them to your search index later")
        else:
            print("⚠️  Skipping search index creation - no markdown files were generated")
        
        # Step 11: RAG preparation information
        print(f"\n🔍 Step 11: RAG Preparation Information")
        print(f"   The generated markdown files are now ready for RAG indexing!")
        print(f"   Search index features:")
        print(f"     • Vector search with Azure OpenAI embeddings")
        print(f"     • Semantic search capabilities")
        print(f"     • Filtering by source file and section")
        print(f"     • Structured format for easy chunking and indexing")
        
        # Step 12: Next steps
        print(f"\n💡 Next Steps for RAG Implementation:")
        print(f"   1. Download markdown files from blob storage")
        print(f"   2. Use the search index for similarity queries")
        print(f"   3. Integrate with your RAG application")
        print(f"   4. Implement query processing and response generation")
        print(f"   5. Add authentication and access controls")
        
        print(f"\n🎉 HEALRAG sample pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 