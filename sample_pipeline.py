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
        supported_types = content_manager.get_supported_file_types()
        supported_files = [f for f in all_files if Path(f).suffix.lower() in supported_types]
        print(f"   Supported files: {len(supported_files)}")
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
        
        # Step 8: RAG preparation information
        print(f"\n🔍 Step 8: RAG Preparation Information")
        print(f"   The generated markdown files are now ready for RAG indexing!")
        print(f"   Each file contains:")
        print(f"     • Original document content")
        print(f"     • Extracted images with descriptions (if Azure OpenAI configured)")
        print(f"     • Image metadata (page numbers, file sizes)")
        print(f"     • Document metadata (source, extraction timestamp)")
        print(f"     • Structured format for easy chunking and indexing")
        
        # Step 9: Next steps
        print(f"\n💡 Next Steps for RAG Implementation:")
        print(f"   1. Download markdown files from blob storage")
        print(f"   2. Chunk the content into appropriate sizes")
        print(f"   3. Create embeddings for each chunk")
        print(f"   4. Store in your vector database")
        print(f"   5. Implement retrieval and generation logic")
        
        print(f"\n🎉 HEALRAG sample pipeline completed successfully!")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        logging.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 