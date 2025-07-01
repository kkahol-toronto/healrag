#!/usr/bin/env python3
"""
HEALRAG Content Extraction Example

This script demonstrates the advanced content extraction capabilities
including image extraction and markdown generation.
"""

import os
import logging
from pathlib import Path

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
    """Main example function."""
    
    # Configuration
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    container_name = os.getenv("AZURE_CONTAINER_NAME", "healrag-documents")
    
    # Azure OpenAI configuration (optional but recommended for image analysis)
    azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
    azure_openai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    
    if not connection_string:
        print("❌ Please set AZURE_STORAGE_CONNECTION_STRING environment variable")
        return
    
    try:
        # Initialize Storage Manager
        print("🚀 Initializing HEALRAG Storage Manager...")
        storage_manager = StorageManager(connection_string, container_name)
        
        if not storage_manager.verify_connection():
            print("❌ Failed to connect to Azure Blob Storage")
            return
        
        print("✅ Connected to Azure Blob Storage")
        
        # Initialize Content Manager
        print("\n📚 Initializing Content Manager...")
        content_manager = ContentManager(
            storage_manager=storage_manager,
            azure_openai_endpoint=azure_openai_endpoint,
            azure_openai_key=azure_openai_key,
            azure_openai_deployment=azure_openai_deployment
        )
        
        # Get file list
        print("\n📄 Getting file list...")
        file_list = storage_manager.get_file_list(as_json=False)
        print(f"Found {len(file_list)} files in container")
        
        if not file_list:
            print("ℹ️  No files found in container. Please upload some documents first.")
            return
        
        # Filter supported files
        supported_types = content_manager.get_supported_file_types()
        supported_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in supported_types)]
        
        print(f"\n📋 Found {len(supported_files)} supported files:")
        for i, file_name in enumerate(supported_files[:10], 1):  # Show first 10
            print(f"   {i}. {file_name}")
        
        if len(supported_files) > 10:
            print(f"   ... and {len(supported_files) - 10} more files")
        
        # Show image extraction support
        print(f"\n🖼️  Image extraction support:")
        image_types = content_manager.get_image_extraction_support()
        print(f"   Supported for: {', '.join(image_types)}")
        
        # Process files with image extraction
        if supported_files:
            print(f"\n🔍 Processing files with content extraction...")
            
            # Custom image prompt for better analysis
            image_prompt = """
            Analyze this image in detail and provide a comprehensive description that includes:
            1. Any text, numbers, or labels visible in the image
            2. Charts, graphs, diagrams, or visual data representations
            3. Images, icons, or visual elements
            4. Layout and structure of the content
            5. Any important context or meaning conveyed by the image
            
            Focus on information that would be valuable for understanding the document content.
            """
            
            # Process files (limit to first 3 for demo)
            files_to_process = supported_files[:3]
            print(f"Processing {len(files_to_process)} files...")
            
            results = content_manager.extract_content_from_files(
                file_list=files_to_process,
                output_folder="md_files",
                image_prompt=image_prompt,
                extract_images=True
            )
            
            # Display results
            print(f"\n📊 Processing Results:")
            successful = 0
            total_images = 0
            
            for file_path, result in results.items():
                if result["success"]:
                    successful += 1
                    total_images += result["images_processed"]
                    print(f"✅ {file_path}")
                    print(f"   📄 Markdown: {result['markdown_file']}")
                    print(f"   🖼️  Images: {result['images_processed']}")
                    print(f"   📏 Length: {result['content_length']:,} characters")
                else:
                    print(f"❌ {file_path}: {result.get('error', 'Unknown error')}")
            
            print(f"\n🎯 Summary:")
            print(f"   Files processed: {successful}/{len(files_to_process)}")
            print(f"   Total images extracted: {total_images}")
            print(f"   Markdown files created in: md_files/")
            
            # Show generated markdown files
            if successful > 0:
                print(f"\n📁 Generated markdown files:")
                md_files = [f for f in storage_manager.get_file_list(as_json=False) if f.startswith("md_files/")]
                for md_file in md_files:
                    print(f"   📄 {md_file}")
        
        # Example: Process specific file types
        print(f"\n🔬 Advanced Processing Examples:")
        
        # PDF files with images
        pdf_files = [f for f in supported_files if f.lower().endswith('.pdf')]
        if pdf_files:
            print(f"   📄 PDF files found: {len(pdf_files)}")
            print(f"   Example: {pdf_files[0]}")
        
        # Word documents
        word_files = [f for f in supported_files if f.lower().endswith(('.docx', '.doc'))]
        if word_files:
            print(f"   📝 Word documents found: {len(word_files)}")
            print(f"   Example: {word_files[0]}")
        
        # PowerPoint presentations
        ppt_files = [f for f in supported_files if f.lower().endswith(('.pptx', '.ppt'))]
        if ppt_files:
            print(f"   📊 PowerPoint files found: {len(ppt_files)}")
            print(f"   Example: {ppt_files[0]}")
        
        print(f"\n🎉 Content extraction example completed!")
        print(f"\n💡 Next steps:")
        print(f"   1. Check the 'md_files' folder in your blob container")
        print(f"   2. Download markdown files for further processing")
        print(f"   3. Use the extracted content for RAG indexing")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Example failed: {e}")

if __name__ == "__main__":
    main() 